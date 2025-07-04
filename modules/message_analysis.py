import json

INPUT_FILE = "groupe_imex_mce_mbt_messages.json"
CHUNK_SIZE = 1000

def chunked_json_array(filename, chunk_size=1000):
    with open(filename, 'r', encoding='utf-8') as f:
        # Read the whole file as a string
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
        # Split objects by '},' (naive but works for pretty-printed arrays)
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
    total = 0
    by_sender_type = {"customer": 0, "community_manager": 0}
    total_length = 0
    for chunk in chunked_json_array(INPUT_FILE, CHUNK_SIZE):
        for msg in chunk:
            total += 1
            stype = msg.get('sender_type', 'unknown')
            by_sender_type[stype] = by_sender_type.get(stype, 0) + 1
            text = msg.get('text', '')
            total_length += len(text)
    print(f"Total messages: {total}")
    for stype, count in by_sender_type.items():
        print(f"{stype}: {count}")
    avg_length = total_length / total if total else 0
    print(f"Average message length: {avg_length:.2f} characters")

if __name__ == "__main__":
    main() 