import json
import csv
import unicodedata
import re

INPUT_FILE = "groupe_imex_mce_mbt_messages.json"
OUTPUT_FILE = "intent_dataset.csv"
CHUNK_SIZE = 1000
MIN_LENGTH = 5  # Minimum message length to include

# Attempt to fix mojibake (mis-encoded unicode)
def fix_mojibake(text):
    try:
        # If text contains common mojibake patterns, try to decode as latin1 then encode as utf-8
        if re.search(r'[Ã©Ã¨ÃªÃ«Ã Ã¢Ã¤Ã¹Ã»Ã¼Ã®Ã¯Ã´Ã¶Ã§Ã‡Ã‰Ã€ÃˆÃ™ÃœÃŒÃŽÃ–Ã"ÃŸ]', text):
            return text.encode('latin1').decode('utf-8')
    except Exception:
        pass
    return text

# Remove non-printable characters
def clean_text(text):
    # Fix mojibake first
    text = fix_mojibake(text)
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    # Remove non-printable characters (except common whitespace)
    text = re.sub(r'[^\x20-\x7E\u00A0-\u017F\u0180-\u024F\u1E00-\u1EFF\s]', '', text)
    # Remove extra invisible characters (zero-width, etc.)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    return text.strip()

def chunked_json_array(filename, chunk_size=1000):
    with open(filename, 'r', encoding='utf-8') as f:
        data = ''
        in_array = False
        for line in f:
            line = line.strip()
            if not in_array:
                if line.startswith('['):
                    in_array = True
                continue
            if line.endswith(']'):
                line = line[:-1]
            if line:
                data += line
        objects = data.split('},')
        buffer = []
        for obj in objects:
            obj = obj.strip()
            if not obj:
                continue
            if not obj.endswith('}'):  # add back the closing brace if missing
                obj += '}'
            try:
                buffer.append(json.loads(obj))
            except Exception:
                continue
            if len(buffer) == chunk_size:
                yield buffer
                buffer = []
        if buffer:
            yield buffer

def main():
    seen_texts = set()
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sender_type', 'text'])
        for chunk in chunked_json_array(INPUT_FILE, CHUNK_SIZE):
            for msg in chunk:
                if msg.get('sender_type') != 'customer':
                    continue
                text = msg.get('text', '').strip()
                if len(text) < MIN_LENGTH:
                    continue
                cleaned = clean_text(text)
                if len(cleaned) < MIN_LENGTH:
                    continue
                if cleaned in seen_texts:
                    continue
                seen_texts.add(cleaned)
                writer.writerow(['customer', cleaned])
    print(f"Intent dataset written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 