# Admin/Correction logic for Messenger bot
import json
import string
import re
import logging
import os
import requests
import time

# Correction/admin state variables
admin_correction_state = {}  # "waiting_for_answer", "waiting_for_correction_mode", "waiting_for_intent", "selecting_intents", "normal"
admin_last_customer_message = {}
admin_selected_intents = {}
CORRECTION_TRIGGERS = ["wrong", "wrong answer", "incorrect", "faux", "erreur", "pas correct"]

# File paths (should be set from app.py)
CORRECTIONS_FILE = os.getenv("CORRECTIONS_FILE", "corrections.jsonl")
LIVE_CORRECTIONS_FILE = os.getenv("LIVE_CORRECTIONS_FILE", "live_corrections.jsonl")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
GRAPH_API_URL = "https://graph.facebook.com/v19.0"

LIVE_CORRECTIONS = {}

# --- Helper functions ---
def strip_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002B50-\U00002B55"  # Stars
        "\U00002300-\U000023FF"  # Misc technical
        "]+",
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_message(text):
    text = text.lower().strip()
    text = strip_emojis(text)
    return text.translate(str.maketrans('', '', string.punctuation))

def load_live_corrections():
    try:
        with open(LIVE_CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                key = normalize_message(entry["customer_message"])
                LIVE_CORRECTIONS[key] = {
                    "intent": entry["correct_intent"],
                    "response": entry["correct_response"]
                }
    except FileNotFoundError:
        pass

def save_live_correction(customer_message, intent, response):
    entry = {
        "customer_message": customer_message,
        "correct_intent": intent,
        "correct_response": response
    }
    with open(LIVE_CORRECTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    key = normalize_message(customer_message)
    LIVE_CORRECTIONS[key] = {"intent": intent, "response": response}

def parse_correction(correction_text):
    try:
        if not correction_text.startswith("CORRECT:"):
            return None
        correction_part = correction_text[8:].strip()
        correction_data = {}
        for item in correction_part.split(','):
            item = item.strip()
            if '=' in item:
                key, value = item.split('=', 1)
                correction_data[key.strip()] = value.strip()
        return correction_data
    except Exception as e:
        logging.error(f"Failed to parse correction: {e}")
        return None

def apply_correction(correction_data, original_message, INTENT_RESPONSES, INTENT_KEYWORDS):
    try:
        intent = correction_data.get('intent')
        response = correction_data.get('response')
        keywords = correction_data.get('keywords', '').split('|') if correction_data.get('keywords') else []
        lang = correction_data.get('lang', 'fr')
        if not (intent and response):
            return False, "Missing intent or response in correction"
        if intent not in INTENT_RESPONSES:
            INTENT_RESPONSES[intent] = {}
        INTENT_RESPONSES[intent][lang] = response
        if keywords:
            if intent not in INTENT_KEYWORDS:
                INTENT_KEYWORDS[intent] = []
            for kw in keywords:
                kw = kw.strip()
                if kw and kw not in INTENT_KEYWORDS[intent]:
                    INTENT_KEYWORDS[intent].append(kw)
        correction_entry = {
            "customer_message": original_message,
            "correct_intent": intent,
            "correct_response": response,
            "new_keywords": keywords,
            "lang": lang,
            "timestamp": time.time()
        }
        with open(CORRECTIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(correction_entry, ensure_ascii=False) + "\n")
        save_live_correction(original_message, intent, response)
        return True, f"Correction applied! Intent: {intent}, Response: {response[:50]}..."
    except Exception as e:
        logging.error(f"Failed to apply correction: {e}")
        return False, f"Error applying correction: {str(e)}"

# --- Messenger UI functions (async) ---
async def send_correction_menu(recipient_id: str):
    """Button-based admin menu (to be called from app.py)"""
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    data = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "button",
                    "text": "What would you like to do?",
                    "buttons": [
                        {"type": "postback", "title": "✅ Correct Intent", "payload": "CORRECT_INTENT_MENU"},
                        {"type": "postback", "title": "📋 View Corrections", "payload": "VIEW_LIVE_CORRECTIONS"},
                        {"type": "postback", "title": "❌ Cancel Correction", "payload": "CANCEL_CORRECTION_FLOW"}
                    ]
                }
            }
        }
    }
    try:
        response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send correction menu: {e}")

async def send_correction_mode_menu(recipient_id: str):
    # Button-based correction mode menu
    pass

async def send_intent_selection_menu(recipient_id, multi_select=False):
    # Button-based intent selection
    pass

# --- New admin features (stubs) ---
async def view_live_corrections(recipient_id: str, page: int = 1):
    """Paginated button-based view of all live corrections (3 per page)"""
    corrections = list(LIVE_CORRECTIONS.items())
    per_page = 3
    total = len(corrections)
    start = (page - 1) * per_page
    end = start + per_page
    page_corrections = corrections[start:end]
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    elements = []
    for key, value in page_corrections:
        elements.append({
            "title": value["intent"],
            "subtitle": value["response"][:80],
            "buttons": [
                {"type": "postback", "title": "🗑 Delete", "payload": f"DELETE_LIVE_CORRECTION|{key}"},
                {"type": "postback", "title": "📜 History", "payload": f"SHOW_CORRECTION_HISTORY|{key}"}
            ]
        })
    # Add navigation if needed
    if total > per_page:
        nav_buttons = []
        if start > 0:
            nav_buttons.append({"type": "postback", "title": "⬅ Prev", "payload": f"VIEW_LIVE_CORRECTIONS|{page-1}"})
        if end < total:
            nav_buttons.append({"type": "postback", "title": "➡ Next", "payload": f"VIEW_LIVE_CORRECTIONS|{page+1}"})
        if nav_buttons:
            elements.append({
                "title": "Navigation",
                "buttons": nav_buttons
            })
    data = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "generic",
                    "elements": elements or [{"title": "No corrections found."}]
                }
            }
        }
    }
    try:
        response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send live corrections view: {e}")

async def delete_live_correction(recipient_id: str, correction_key: str):
    """Delete a live correction by key and confirm to admin"""
    if correction_key in LIVE_CORRECTIONS:
        del LIVE_CORRECTIONS[correction_key]
        # Rewrite file
        with open(LIVE_CORRECTIONS_FILE, "w", encoding="utf-8") as f:
            for k, v in LIVE_CORRECTIONS.items():
                entry = {
                    "customer_message": k,
                    "correct_intent": v["intent"],
                    "correct_response": v["response"]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        msg = "✅ Correction deleted."
    else:
        msg = "❌ Correction not found."
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    data = {"recipient": {"id": recipient_id}, "message": {"text": msg}}
    try:
        response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send delete confirmation: {e}")

async def cancel_correction_flow(recipient_id: str):
    """Cancel any in-progress correction and confirm to admin"""
    admin_correction_state[recipient_id] = "normal"
    admin_selected_intents[recipient_id] = []
    msg = "❌ Correction flow cancelled."
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    data = {"recipient": {"id": recipient_id}, "message": {"text": msg}}
    try:
        response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send cancel confirmation: {e}")

async def show_correction_history(recipient_id: str, customer_message: str):
    """Show all corrections for a specific normalized message (from both live and corrections file)"""
    norm_msg = normalize_message(customer_message)
    history = []
    # From live corrections
    if norm_msg in LIVE_CORRECTIONS:
        history.append(LIVE_CORRECTIONS[norm_msg])
    # From corrections file
    try:
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if normalize_message(entry["customer_message"]) == norm_msg:
                    history.append({"intent": entry["correct_intent"], "response": entry["correct_response"]})
    except Exception:
        pass
    if not history:
        msg = "No correction history found."
    else:
        msg = "\n\n".join([f"Intent: {h['intent']}\nResponse: {h['response'][:80]}" for h in history])
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    data = {"recipient": {"id": recipient_id}, "message": {"text": msg}}
    try:
        response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send correction history: {e}") 