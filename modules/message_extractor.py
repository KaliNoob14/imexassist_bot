import os
import json
import glob
from pathlib import Path

PAGE_NAME = "Groupe IMEX MCE MBT"

# Helper to determine sender type
def get_sender_type(sender_name):
    if sender_name == PAGE_NAME:
        return "community_manager"
    return "customer"

def thread_has_page_participant(thread_path):
    try:
        with open(thread_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            participants = [p.get('name', '') for p in data.get('participants', [])]
            return PAGE_NAME in participants
    except Exception:
        return False

def extract_messages_from_thread(thread_path):
    messages = []
    with open(thread_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for msg in data.get('messages', []):
            sender = msg.get('sender_name', '')
            text = msg.get('content', '')
            if not text:
                continue  # skip non-text messages
            messages.append({
                'sender_type': get_sender_type(sender),
                'sender_name': sender,
                'timestamp_ms': msg.get('timestamp_ms'),
                'text': text
            })
    return messages

def extract_all_messages(messages_folder):
    all_messages = []
    for box in ['inbox', 'filtered_threads']:
        box_path = Path(messages_folder) / box
        if not box_path.exists():
            continue
        for thread_dir in box_path.iterdir():
            if not thread_dir.is_dir():
                continue
            for json_file in thread_dir.glob('message_*.json'):
                if thread_has_page_participant(json_file):
                    all_messages.extend(extract_messages_from_thread(json_file))
    return all_messages

def main():
    messages_folder = input("Enter the path to your Facebook messages folder: ").strip()
    if not os.path.isdir(messages_folder):
        print(f"Error: {messages_folder} is not a valid directory.")
        return
    print(f"Extracting messages for page: {PAGE_NAME}")
    messages = extract_all_messages(messages_folder)
    print(f"Extracted {len(messages)} messages.")
    out_path = os.path.join(os.getcwd(), f"{PAGE_NAME.replace(' ', '_').lower()}_messages.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"Saved extracted messages to {out_path}")

if __name__ == "__main__":
    main() 