# app.py (Simplified for quick test)
from fastapi import FastAPI, Request, Response, HTTPException, Header
from dotenv import load_dotenv
import os
import requests
import logging
import time
from language_nn import detect_language
from intent_nn import predict_intent
from dictionary.intent_keywords import INTENT_KEYWORDS
import json
import string
import re

load_dotenv() # Load environment variables from .env during local dev

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Retrieve these from environment variables set during Cloud Run deployment
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
PAGE_ID = os.getenv("PAGE_ID")  # Add PAGE_ID to environment variables

GRAPH_API_URL = "https://graph.facebook.com/v19.0" # Use a recent API version

# Simple rate limiting - track recent messages per user
recent_messages = {}

CORRECTIONS_FILE = "corrections.jsonl"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "supersecret")  # Set this in your env for security

# Admin sender IDs who can correct the bot (add your Facebook user ID here)
ADMIN_SENDER_IDS = [
    "10073659842756382",  # Add your Facebook user ID here
    # Add more admin IDs as needed
]

LIVE_CORRECTIONS_FILE = "live_corrections.jsonl"
LIVE_CORRECTIONS = {}

def strip_emojis(text):
    # Remove all emoji characters using regex
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
    # Lowercase, strip, remove punctuation, remove emojis
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

# Load live corrections on startup
load_live_corrections()

def is_rate_limited(sender_id: str, message_text: str, cooldown_seconds: int = 3) -> bool:
    """Check if user is sending messages too quickly"""
    current_time = time.time()
    key = f"{sender_id}:{message_text.strip().lower()}"
    
    if key in recent_messages:
        time_diff = current_time - recent_messages[key]
        if time_diff < cooldown_seconds:
            return True
    
    recent_messages[key] = current_time
    return False

# Function to send a text message (asynchronous)
async def send_text_message(recipient_id: str, message_text: str):
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    data = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    try:
        response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        logging.info(f"Message sent successfully to {recipient_id}: {message_text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending message to {recipient_id}: {e}")
        if response is not None:
            logging.error(f"Response: {response.text}")

# Webhook Verification Endpoint
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    # --- DEBUGGING PRINTS ---
    logging.info(f"[DEBUG] /webhook GET called")
    logging.info(f"[DEBUG] Received mode: {mode}")
    logging.info(f"[DEBUG] Received token: '{token}' (length: {len(token) if token else 0})")
    logging.info(f"[DEBUG] Expected token: '{VERIFY_TOKEN}' (length: {len(VERIFY_TOKEN) if VERIFY_TOKEN else 0})")
    logging.info(f"[DEBUG] Tokens match check: {token == VERIFY_TOKEN}")
    # --- END DEBUGGING PRINTS ---

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logging.info("[DEBUG] WEBHOOK_VERIFIED")
            return Response(content=challenge, status_code=200)
        else:
            logging.info("[DEBUG] Verification token mismatch")
            return Response(content="Verification token mismatch", status_code=403)
    logging.info("[DEBUG] Missing parameters")
    return Response(content="Missing parameters", status_code=400)

# Webhook Message Handling Endpoint
@app.post("/webhook")
async def handle_message(request: Request):
    data = await request.json()
    print(f"[DEBUG] Received webhook data: {data}")

    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                sender_id = messaging_event["sender"]["id"]
                
                # Skip if message is from the page itself (prevents infinite loop)
                if sender_id == PAGE_ID:
                    print(f"[DEBUG] Skipping message from page itself: {sender_id}")
                    continue

                # Skip delivery receipts, read receipts, and other non-message events
                if "delivery" in messaging_event or "read" in messaging_event:
                    print(f"[DEBUG] Skipping delivery/read receipt from {sender_id}")
                    continue

                # Handle text messages
                if "message" in messaging_event and "text" in messaging_event["message"]:
                    message_text = messaging_event["message"]["text"]
                    print(f"[DEBUG] Received message from {sender_id}: {message_text}")

                    # --- ADMIN HANDLING ---
                    if sender_id in ADMIN_SENDER_IDS:
                        logging.info(f"[ADMIN DEBUG] sender_id: {sender_id}, ADMIN_SENDER_IDS: {ADMIN_SENDER_IDS}")
                        # --- ADMIN MENU TRIGGER (must be first) ---
                        if message_text.lower().strip() in ADMIN_MENU_TRIGGERS:
                            logging.info(f"[ADMIN MENU TRIGGER] Trigger word received from {sender_id}: {message_text}")
                            await send_correction_menu(sender_id)
                            return Response(content="OK", status_code=200)
                        # --- Handle pending delete flow ---
                        if sender_id in admin_pending_delete and admin_pending_delete[sender_id]:
                            # Try to delete the normalized message from live corrections
                            norm_msg = normalize_message(message_text)
                            if norm_msg in LIVE_CORRECTIONS:
                                # Remove from memory
                                del LIVE_CORRECTIONS[norm_msg]
                                # Remove from file
                                try:
                                    with open(LIVE_CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                                        lines = f.readlines()
                                    with open(LIVE_CORRECTIONS_FILE, "w", encoding="utf-8") as f:
                                        for line in lines:
                                            entry = json.loads(line)
                                            if normalize_message(entry["customer_message"]) != norm_msg:
                                                f.write(line)
                                    reply = "✅ Live correction deleted."
                                except Exception as e:
                                    reply = f"❌ Error deleting: {e}"
                            else:
                                reply = "❌ No live correction found for that message."
                            await send_text_message(sender_id, reply)
                            admin_pending_delete[sender_id] = False
                            return Response(content="OK", status_code=200)
                        # --- Handle pending history flow ---
                        if sender_id in admin_pending_history and admin_pending_history[sender_id]:
                            norm_msg = normalize_message(message_text)
                            # Search corrections.jsonl for all corrections for this normalized message
                            history = []
                            try:
                                with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                                    for line in f:
                                        entry = json.loads(line)
                                        if normalize_message(entry["customer_message"]) == norm_msg:
                                            history.append(entry)
                            except Exception as e:
                                await send_text_message(sender_id, f"❌ Error reading history: {e}")
                                admin_pending_history[sender_id] = False
                                return Response(content="OK", status_code=200)
                            if not history:
                                await send_text_message(sender_id, "No correction history found for that message.")
                            else:
                                for entry in history[-5:]:  # Show up to last 5
                                    ts = entry.get("timestamp")
                                    ts_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts)) if ts else ""
                                    msg = f"[{ts_str}] Intent: {entry.get('correct_intent')}\nResp: {entry.get('correct_response')}"
                                    await send_text_message(sender_id, msg)
                            admin_pending_history[sender_id] = False
                            return Response(content="OK", status_code=200)
                    # --- END ADMIN HANDLING ---

                    # --- ADMIN CORRECTION FLOW (handle before normal message processing) ---
                    if sender_id in ADMIN_SENDER_IDS:
                        current_state = admin_correction_state.get(sender_id, "normal")
                        # Correction step: waiting for correct answer
                        if current_state == "waiting_for_answer":
                            admin_last_customer_message[sender_id] = message_text
                            admin_correction_state[sender_id] = "waiting_for_correction_mode"
                            await send_correction_mode_menu(sender_id)
                            return Response(content="OK", status_code=200)
                        # Correction step: waiting for correction mode (single/multi intent)
                        elif current_state == "waiting_for_correction_mode":
                            # Only handle via postback, ignore text
                            return Response(content="OK", status_code=200)
                        # Correction step: waiting for intent selection (handled in postback)
                        elif current_state in ("waiting_for_intent", "selecting_intents"):
                            # Do nothing here, handled in postback
                            return Response(content="OK", status_code=200)
                        # Correction trigger
                        if message_text.lower().strip() in CORRECTION_TRIGGERS:
                            admin_correction_state[sender_id] = "waiting_for_answer"
                            reply = "Teach me the correct answer for the last customer message:"
                            await send_text_message(sender_id, reply)
                            return Response(content="OK", status_code=200)
                        # Manual correction command
                        correction_data = parse_correction(message_text)
                        if correction_data:
                            success, message = apply_correction(correction_data, message_text)
                            reply = f"✅ {message}" if success else f"❌ {message}"
                            await send_text_message(sender_id, reply)
                            admin_correction_state[sender_id] = "normal"
                            return Response(content="OK", status_code=200)
                        # Not in correction mode, store message for potential correction, but process as normal message
                        admin_last_customer_message[sender_id] = message_text
                    # --- END ADMIN CORRECTION FLOW ---

                    # --- Handle quick reply payloads for intent selection ---
                    if "quick_reply" in messaging_event["message"]:
                        quick_reply_payload = messaging_event["message"]["quick_reply"]["payload"]
                        # Reuse the postback logic for intent selection
                        if sender_id in ADMIN_SENDER_IDS and quick_reply_payload.startswith("INTENT_"):
                            selected_intent = quick_reply_payload.replace("INTENT_", "")
                            last_message = admin_last_customer_message.get(sender_id, "No message stored")
                            current_state = admin_correction_state.get(sender_id, "normal")
                            if current_state not in ("waiting_for_intent", "selecting_intents"):
                                await send_text_message(sender_id, "No correction in progress. Please trigger correction first.")
                                admin_correction_state[sender_id] = "normal"
                                admin_selected_intents[sender_id] = []
                                return Response(content="OK", status_code=200)
                            if selected_intent == "DONE":
                                selected_intents = admin_selected_intents.get(sender_id, [])
                                if not selected_intents:
                                    reply = "❌ No intents selected. Please select at least one intent."
                                    await send_text_message(sender_id, reply)
                                    admin_correction_state[sender_id] = "normal"
                                    admin_selected_intents[sender_id] = []
                                    return Response(content="OK", status_code=200)
                                correct_answer = last_message
                                success_count = 0
                                for intent in selected_intents:
                                    correction_data = {
                                        "intent": intent,
                                        "response": correct_answer,
                                        "lang": "fr"
                                    }
                                    success, _ = apply_correction(correction_data, last_message)
                                    if success:
                                        success_count += 1
                                reply = f"✅ Applied correction to {success_count}/{len(selected_intents)} intents: {', '.join(selected_intents)}"
                                await send_text_message(sender_id, reply)
                                # Always reset state after correction
                                admin_correction_state[sender_id] = "normal"
                                admin_selected_intents[sender_id] = []
                                return Response(content="OK", status_code=200)
                            if current_state == "selecting_intents":
                                if sender_id not in admin_selected_intents:
                                    admin_selected_intents[sender_id] = []
                                if selected_intent not in admin_selected_intents[sender_id]:
                                    admin_selected_intents[sender_id].append(selected_intent)
                                    reply = f"✅ Added: {selected_intent}\nSelected: {', '.join(admin_selected_intents[sender_id])}\nTap ✅ Done when finished."
                                else:
                                    reply = f"⚠️ {selected_intent} already selected.\nSelected: {', '.join(admin_selected_intents[sender_id])}\nTap ✅ Done when finished."
                                await send_text_message(sender_id, reply)
                                return Response(content="OK", status_code=200)
                            correct_answer = last_message
                            correction_data = {
                                "intent": selected_intent,
                                "response": correct_answer,
                                "lang": "fr"
                            }
                            success, message = apply_correction(correction_data, last_message)
                            reply = f"✅ {message}" if success else f"❌ {message}"
                            await send_text_message(sender_id, reply)
                            # Always reset state after correction
                            admin_correction_state[sender_id] = "normal"
                            admin_selected_intents[sender_id] = []
                            return Response(content="OK", status_code=200)
                    # --- END Handle quick reply payloads for intent selection ---

                    # --- Check for empty or whitespace message ---
                    if not message_text or not message_text.strip():
                        print(f"[DEBUG] Empty or whitespace message received from {sender_id}")
                        reply = "Sorry, I couldn't detect your language."
                        await send_text_message(sender_id, reply)
                        continue

                    # --- LIVE CORRECTION CHECK ---
                    norm_msg = normalize_message(message_text)
                    if norm_msg in LIVE_CORRECTIONS:
                        corr = LIVE_CORRECTIONS[norm_msg]
                        reply = corr["response"]
                        await send_text_message(sender_id, reply)
                        continue
                    # --- STRIP EMOJIS FOR INTENT RECOGNITION ---
                    clean_message_text = strip_emojis(message_text)
                    # --- Rate limiting check ---
                    if is_rate_limited(sender_id, message_text):
                        print(f"[DEBUG] Rate limited message from {sender_id}: {message_text}")
                        continue
                    # --- fastText language detection and response ---
                    lang_code, confidence = detect_language(clean_message_text)
                    print(f"[DEBUG] fastText detected: {lang_code} (confidence: {confidence:.2f})")
                    # --- Intent detection ---
                    intent, intent_conf = predict_intent(clean_message_text)
                    print(f"[DEBUG] Predicted intent: {intent} (confidence: {intent_conf:.2f})")

                    # --- Intent-based response logic ---
                    reply_lang = lang_code if lang_code in ["fr", "mg", "en"] else "en"

                    # Special case: exact 'Bonsoir' message
                    if message_text.strip().lower() == "bonsoir":
                        reply = "Bonsoir ! Nous sommes là pour vous aider. Comment pouvons-nous vous assister aujourd'hui ?"
                    else:
                        # 1. Use neural network intent if confident
                        if intent and intent_conf >= 0.5 and intent in INTENT_RESPONSES:
                            if intent == "pricing":
                                location = extract_location(message_text)
                                if location == "china":
                                    reply = PRICING_CONTACTS["china"][reply_lang]
                                elif location == "indonesia":
                                    reply = PRICING_CONTACTS["indonesia"][reply_lang]
                                elif location in {"thailand", "canada", "france", "uae"}:
                                    reply = PRICING_CONTACTS["other"][reply_lang]
                                else:
                                    reply = INTENT_RESPONSES["pricing"][reply_lang]
                            else:
                                reply = INTENT_RESPONSES[intent][reply_lang]
                        else:
                            # 2. Fallback: keyword matching
                            matched_intent = None
                            text_lower = message_text.lower()
                            for k_intent, keywords in INTENT_KEYWORDS.items():
                                if any(kw in text_lower for kw in keywords):
                                    matched_intent = k_intent
                                    break
                            if matched_intent:
                                if matched_intent == "pricing":
                                    location = extract_location(message_text)
                                    if location == "china":
                                        reply = PRICING_CONTACTS["china"][reply_lang]
                                    elif location == "indonesia":
                                        reply = PRICING_CONTACTS["indonesia"][reply_lang]
                                    elif location in {"thailand", "canada", "france", "uae"}:
                                        reply = PRICING_CONTACTS["other"][reply_lang]
                                    else:
                                        reply = INTENT_RESPONSES["pricing"][reply_lang]
                                else:
                                    reply = INTENT_RESPONSES[matched_intent][reply_lang]
                            else:
                                reply = DEFAULT_RESPONSES[reply_lang]
                    await send_text_message(sender_id, reply)
                    # --- End intent-based response logic ---

                # Handle postback from buttons/persistent menu
                elif "postback" in messaging_event:
                    payload = messaging_event["postback"]["payload"]
                    print(f"[DEBUG] Received postback from {sender_id}: {payload}")
                    
                    # Handle admin postbacks
                    if sender_id in ADMIN_SENDER_IDS:
                        if payload == "CORRECT_INTENT_MENU":
                            # Instead of sending quick replies, call send_correction_mode_menu(recipient_id)
                            await send_correction_mode_menu(sender_id)
                            continue
                        elif payload == "VIEW_STATS":
                            # Count corrections in the file
                            try:
                                with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
                                    correction_count = sum(1 for line in f)
                                reply = f"📊 Bot Statistics:\n- Total corrections: {correction_count}\n- Active intents: {len(INTENT_RESPONSES)}"
                            except FileNotFoundError:
                                reply = "📊 Bot Statistics:\n- Total corrections: 0\n- Active intents: 0"
                            await send_text_message(sender_id, reply)
                            continue
                        elif payload == "ADMIN_HELP":
                            help_text = """🔧 Admin Commands:
• Use the menu to correct intents
• Type: CORRECT: intent=X, response=Y, keywords=Z
• View stats and get help via menu"""
                            await send_text_message(sender_id, help_text)
                            continue
                        elif payload.startswith("INTENT_"):
                            # Handle intent selection
                            selected_intent = payload.replace("INTENT_", "")
                            last_message = admin_last_customer_message.get(sender_id, "No message stored")
                            
                            # Only handle intent selection if admin is in correction flow
                            current_state = admin_correction_state.get(sender_id, "normal")
                            if current_state not in ("waiting_for_intent", "selecting_intents"):
                                # Ignore intent selection if not in correction flow
                                await send_text_message(sender_id, "No correction in progress. Please trigger correction first.")
                                return Response(content="OK", status_code=200)
                            
                            if selected_intent == "DONE":
                                # Finalize multi-intent selection
                                selected_intents = admin_selected_intents.get(sender_id, [])
                                if not selected_intents:
                                    reply = "❌ No intents selected. Please select at least one intent."
                                    await send_text_message(sender_id, reply)
                                    admin_correction_state[sender_id] = "normal"
                                    admin_selected_intents[sender_id] = []
                                    return Response(content="OK", status_code=200)
                                
                                # Get the correct answer from the stored message
                                correct_answer = last_message
                                
                                # Apply correction for each selected intent
                                success_count = 0
                                for intent in selected_intents:
                                    correction_data = {
                                        "intent": intent,
                                        "response": correct_answer,
                                        "lang": "fr"
                                    }
                                    success, _ = apply_correction(correction_data, last_message)
                                    if success:
                                        success_count += 1
                                
                                reply = f"✅ Applied correction to {success_count}/{len(selected_intents)} intents: {', '.join(selected_intents)}"
                                await send_text_message(sender_id, reply)
                                
                                # Reset correction state
                                admin_correction_state[sender_id] = "normal"
                                admin_selected_intents[sender_id] = []
                                return Response(content="OK", status_code=200)
                            
                            # Check if we're in multi-select mode
                            if current_state == "selecting_intents":
                                # Add to selected intents
                                if sender_id not in admin_selected_intents:
                                    admin_selected_intents[sender_id] = []
                                
                                if selected_intent not in admin_selected_intents[sender_id]:
                                    admin_selected_intents[sender_id].append(selected_intent)
                                    reply = f"✅ Added: {selected_intent}\nSelected: {', '.join(admin_selected_intents[sender_id])}\nTap ✅ Done when finished."
                                else:
                                    reply = f"⚠️ {selected_intent} already selected.\nSelected: {', '.join(admin_selected_intents[sender_id])}\nTap ✅ Done when finished."
                                
                                await send_text_message(sender_id, reply)
                                return Response(content="OK", status_code=200)
                            
                            # Single intent selection (original behavior)
                            # Get the correct answer from the stored message
                            correct_answer = last_message
                            
                            # Apply the correction
                            correction_data = {
                                "intent": selected_intent,
                                "response": correct_answer,
                                "lang": "fr"
                            }
                            success, message = apply_correction(correction_data, last_message)
                            reply = f"✅ {message}" if success else f"❌ {message}"
                            await send_text_message(sender_id, reply)
                            
                            # Reset correction state
                            admin_correction_state[sender_id] = "normal"
                            return Response(content="OK", status_code=200)
                        elif payload == "SINGLE_INTENT":
                            admin_correction_state[sender_id] = "waiting_for_intent"
                            await send_intent_selection_menu(sender_id, multi_select=False)
                            continue
                        elif payload == "MULTI_INTENT":
                            admin_correction_state[sender_id] = "selecting_intents"
                            admin_selected_intents[sender_id] = []
                            await send_intent_selection_menu(sender_id, multi_select=True)
                            continue
                        elif payload == "VIEW_LIVE_CORRECTIONS":
                            # List all live corrections (show up to 10)
                            if not LIVE_CORRECTIONS:
                                await send_text_message(sender_id, "No live corrections saved.")
                            else:
                                count = 0
                                for k, v in list(LIVE_CORRECTIONS.items()):
                                    if count >= 10:
                                        await send_text_message(sender_id, f"...and {len(LIVE_CORRECTIONS)-10} more.")
                                        break
                                    await send_text_message(sender_id, f"Msg: {k}\nIntent: {v['intent']}\nResp: {v['response']}")
                                    count += 1
                            continue
                        elif payload == "DELETE_LIVE_CORRECTION":
                            admin_pending_delete[sender_id] = True
                            await send_text_message(sender_id, "Send the customer message to delete its live correction.")
                            continue
                        elif payload == "CANCEL_CORRECTION_FLOW":
                            admin_correction_state[sender_id] = "normal"
                            admin_selected_intents[sender_id] = []
                            admin_pending_delete[sender_id] = False
                            admin_pending_history[sender_id] = False
                            await send_text_message(sender_id, "Correction flow cancelled and state reset.")
                            continue
                        elif payload == "SHOW_CORRECTION_HISTORY":
                            admin_pending_history[sender_id] = True
                            await send_text_message(sender_id, "Send the customer message to view its correction history.")
                            continue
                    
                    # Handle regular postbacks
                    await send_text_message(sender_id, f"Received payload: {payload}")

    return Response(content="OK", status_code=200)

# You can add a simple root endpoint for health checks or general info
@app.get("/")
async def root():
    logging.info("Root endpoint hit!")
    return {"message": "Hello, I'm IMEX Assist, how may I help you?"}

# Intent to response mapping
INTENT_RESPONSES = {
    "greeting": {
        "en": "Hello! We are here to help you. How can we assist you today?",
        "fr": "Bonjour ! Nous sommes là pour vous aider. Comment pouvons-nous vous assister aujourd'hui ?",
        "mg": "Salama! Eto izahay hanampy anao. Inona no azontsika atao ho anao androany?"
    },
    "thanks": {
        "fr": "Nous vous remercions pour votre confiance. N'hésitez pas à nous recontacter si vous avez d'autres questions !",
        "en": "Thank you for your trust. Don't hesitate to contact us again if you have any other questions!",
        "mg": "Misaotra amin'ny fahatokisanao anay. Aza misalasala raha manana fanontaniana hafa ianao!"
    },
    "pricing": {
        "en": "For pricing, please discuss with our sales team.",
        "fr": "Concernant les tarifs, il faut discuter avec l'équipe commerciale monsieur/madame",
        "mg": "Momba ny vidiny, azafady mifandraisa amin'ny ekipan'ny varotra."
    },
    "shipping_duration": {
        "en": "Our sea delivery time is 45 to 60 days.",
        "fr": "Notre délai de livraison maritime est de 45 à 60 jours.",
        "mg": "Ny faharetan'ny fandefasana an-dranomasina dia 45 ka hatramin'ny 60 andro."
    },
    "shipping": {
        "fr": "La livraison dépend de votre emplacement et du service choisi.",
        "mg": "Miankina amin'ny toerana sy ny tolotra no fandefasana.",
        "en": "Shipping depends on your location and the chosen service."
    },
    "opening_hours": {
        "fr": "Nos horaires d'ouverture sont de 8h à 17h, du lundi au vendredi.",
        "mg": "Misokatra 8 ora maraina ka hatramin'ny 5 ora hariva izahay, alatsinainy ka hatramin'ny zoma.",
        "en": "We are open from 8am to 5pm, Monday to Friday."
    },
    "contact": {
        "fr": "Vous pouvez nous contacter au 034 12 345 67 ou par email à info@imex.com.",
        "mg": "Afaka miantso anay amin'ny 034 12 345 67 na manoratra amin'ny info@imex.com ianao.",
        "en": "You can contact us at 034 12 345 67 or by email at info@imex.com."
    },
    "location": {
        "fr": "Notre bureau se trouve à Antananarivo, Lot II F 23.",
        "mg": "Any Antananarivo, Lot II F 23 ny biraonay.",
        "en": "Our office is in Antananarivo, Lot II F 23."
    },
    "product_info": {
        "fr": "Nous proposons divers produits et services. Voulez-vous plus de détails?",
        "mg": "Manolotra vokatra sy tolotra maro izahay. Mila fanazavana fanampiny ve ianao?",
        "en": "We offer various products and services. Would you like more details?"
    }
}
DEFAULT_RESPONSES = {
    "fr": "Je n'ai pas compris votre demande. Pouvez-vous préciser?",
    "mg": "Tsy azoko tsara ny fangatahanao. Azafady hazavao.",
    "en": "I didn't understand your request. Could you clarify?"
}

# Location keywords for context extraction
LOCATION_KEYWORDS = {
    "china": ["chine", "china"],
    "thailand": ["thaïlande", "thailand"],
    "canada": ["canada"],
    "france": ["france"],
    "uae": ["uae", "emirats", "emirates", "dubai"],
    "indonesia": ["indonésie", "indonesia"]
}

# Location-specific pricing contacts
PRICING_CONTACTS = {
    "china": {
        "fr": (
            "Concernant les tarifs, il faut discuter avec l'équipe commerciale madame/monsieur. "
            "Voici leurs numéros de téléphone, WhatsApp et WeChat : Chine :\n"
            "🇨🇳 Mme Hasina : 034 05 828 71\n"
            "🇨🇳 Mme Malala : 034 05 828 72\n"
            "🇨🇳 Mme Bodo : 034 05 828 73"
        ),
        "en": (
            "For pricing, please contact our sales team. Here are their phone, WhatsApp, and WeChat numbers for China:\n"
            "🇨🇳 Ms. Hasina: 034 05 828 71\n"
            "🇨🇳 Ms. Malala: 034 05 828 72\n"
            "🇨🇳 Ms. Bodo: 034 05 828 73"
        ),
        "mg": (
            "Momba ny vidiny, azafady mifandraisa amin'ny ekipan'ny varotra. Ireto ny laharan'ny ekipanay any Chine:\n"
            "🇨🇳 Mme Hasina : 034 05 828 71\n"
            "🇨🇳 Mme Malala : 034 05 828 72\n"
            "🇨🇳 Mme Bodo : 034 05 828 73"
        )
    },
    "indonesia": {
        "fr": "Concernant les tarifs, contactez Mme Natasy : 034 05 828 96 pour l'Indonésie.",
        "en": "For pricing, please contact Ms. Natasy: 034 05 828 96 for Indonesia.",
        "mg": "Momba ny vidiny, antsoy i Mme Natasy: 034 05 828 96 ho an'i Indonezia."
    },
    "other": {
        "fr": (
            "Concernant les tarifs, il faut discuter avec l'équipe commerciale madame/monsieur. "
            "Voici leur numéro de téléphone, WhatsApp et WeChat :\n"
            "Mme Annie : 034 05 828 87"
        ),
        "en": (
            "For pricing, please contact our sales team. Here is their phone, WhatsApp, and WeChat number:\n"
            "Ms. Annie: 034 05 828 87"
        ),
        "mg": (
            "Momba ny vidiny, azafady mifandraisa amin'ny ekipan'ny varotra. Ireto ny laharan'ny ekipanay:\n"
            "Mme Annie : 034 05 828 87"
        )
    }
}

# Function to extract location from message text
def extract_location(message_text):
    text = message_text.lower()
    for loc, keywords in LOCATION_KEYWORDS.items():
        if any(k in text for k in keywords):
            return loc
    return None

def parse_correction(correction_text):
    """
    Parse admin correction message in format: 
    "CORRECT: intent=X, response=Y, keywords=Z"
    """
    try:
        if not correction_text.startswith("CORRECT:"):
            return None
        
        # Extract the correction part after "CORRECT:"
        correction_part = correction_text[8:].strip()
        
        # Parse key-value pairs
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

def apply_correction(correction_data, original_message):
    """
    Apply correction to bot's memory and save for retraining
    """
    try:
        intent = correction_data.get('intent')
        response = correction_data.get('response')
        keywords = correction_data.get('keywords', '').split('|') if correction_data.get('keywords') else []
        lang = correction_data.get('lang', 'fr')
        
        if not (intent and response):
            return False, "Missing intent or response in correction"
        
        # Update INTENT_RESPONSES in memory
        if intent not in INTENT_RESPONSES:
            INTENT_RESPONSES[intent] = {}
        INTENT_RESPONSES[intent][lang] = response
        
        # Update INTENT_KEYWORDS in memory
        if keywords:
            if intent not in INTENT_KEYWORDS:
                INTENT_KEYWORDS[intent] = []
            for kw in keywords:
                kw = kw.strip()
                if kw and kw not in INTENT_KEYWORDS[intent]:
                    INTENT_KEYWORDS[intent].append(kw)
        
        # Save correction for retraining
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
        
        # --- LIVE CORRECTION PERSISTENCE ---
        save_live_correction(original_message, intent, response)

        return True, f"Correction applied! Intent: {intent}, Response: {response[:50]}..."
        
    except Exception as e:
        logging.error(f"Failed to apply correction: {e}")
        return False, f"Error applying correction: {str(e)}"

@app.post("/correction")
async def correction_endpoint(
    request: Request,
    authorization: str = Header(None)
):
    # Simple admin token check
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = await request.json()
    customer_message = data.get("customer_message")
    correct_intent = data.get("correct_intent")
    correct_response = data.get("correct_response")
    new_keywords = data.get("new_keywords", [])
    lang = data.get("lang", "fr")  # Default to French if not specified
    if not (customer_message and correct_intent and correct_response):
        raise HTTPException(status_code=400, detail="Missing required fields.")
    # Update INTENT_RESPONSES in memory
    if correct_intent not in INTENT_RESPONSES:
        INTENT_RESPONSES[correct_intent] = {}
    INTENT_RESPONSES[correct_intent][lang] = correct_response
    # Update INTENT_KEYWORDS in memory
    if new_keywords:
        if correct_intent not in INTENT_KEYWORDS:
            INTENT_KEYWORDS[correct_intent] = []
        for kw in new_keywords:
            if kw not in INTENT_KEYWORDS[correct_intent]:
                INTENT_KEYWORDS[correct_intent].append(kw)
    # Save correction for retraining
    correction_entry = {
        "customer_message": customer_message,
        "correct_intent": correct_intent,
        "correct_response": correct_response,
        "new_keywords": new_keywords,
        "lang": lang
    }
    with open(CORRECTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(correction_entry, ensure_ascii=False) + "\n")
    # --- LIVE CORRECTION PERSISTENCE ---
    save_live_correction(customer_message, correct_intent, correct_response)
    return {"status": "success", "message": "Correction applied and saved."}

async def setup_persistent_menu():
    """Set up persistent menu for admin corrections"""
    menu_data = {
        "persistent_menu": [
            {
                "locale": "default",
                "composer_input_disabled": False,
                "call_to_actions": [
                    {
                        "type": "postback",
                        "title": "🔧 Correct Intent",
                        "payload": "CORRECT_INTENT_MENU"
                    },
                    {
                        "type": "postback", 
                        "title": "📊 View Stats",
                        "payload": "VIEW_STATS"
                    },
                    {
                        "type": "postback",
                        "title": "❓ Help",
                        "payload": "ADMIN_HELP"
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{GRAPH_API_URL}/me/messenger_profile",
            params={"access_token": PAGE_ACCESS_TOKEN},
            json=menu_data
        )
        response.raise_for_status()
        logging.info("Persistent menu set up successfully")
    except Exception as e:
        logging.error(f"Failed to set up persistent menu: {e}")

async def send_intent_selection_menu(recipient_id, multi_select=False):
    """Send a button template for intent selection (single or multi-select)"""
    # Define all intents as button template buttons
    intent_buttons = [
        {"type": "postback", "title": "👋 Greeting", "payload": "INTENT_greeting"},
        {"type": "postback", "title": "💰 Pricing", "payload": "INTENT_pricing"},
        {"type": "postback", "title": "🚢 Shipping", "payload": "INTENT_shipping"},
        {"type": "postback", "title": "⏰ Duration", "payload": "INTENT_shipping_duration"},
        {"type": "postback", "title": "📞 Contact", "payload": "INTENT_contact"},
        {"type": "postback", "title": "📍 Location", "payload": "INTENT_location"},
        {"type": "postback", "title": "🛍️ Products", "payload": "INTENT_product_info"},
        {"type": "postback", "title": "🙏 Thanks", "payload": "INTENT_thanks"},
        {"type": "postback", "title": "🕐 Hours", "payload": "INTENT_opening_hours"}
    ]
    message_text = "Select the correct intent for the last customer message:"
    if multi_select:
        # Add 'Done' button to the last template
        intent_buttons.append({"type": "postback", "title": "✅ Done", "payload": "INTENT_DONE"})
        message_text = "Select all relevant intents (tap ✅ Done when finished):"

    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    # Send button templates in groups of 3
    for i in range(0, len(intent_buttons), 3):
        buttons = intent_buttons[i:i+3]
        data = {
            "recipient": {"id": recipient_id},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "button",
                        "text": message_text if i == 0 else "More intents:",
                        "buttons": buttons
                    }
                }
            }
        }
        try:
            response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
            response.raise_for_status()
        except Exception as e:
            logging.error(f"Failed to send intent button template: {e}")

# --- REPLACE correction mode quick replies with button template ---
async def send_correction_mode_menu(recipient_id):
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    data = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "button",
                    "text": "Choose correction mode:",
                    "buttons": [
                        {"type": "postback", "title": "🎯 Single Intent", "payload": "SINGLE_INTENT"},
                        {"type": "postback", "title": "🎯🎯 Multiple Intents", "payload": "MULTI_INTENT"}
                    ]
                }
            }
        }
    }
    try:
        response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to send correction mode menu: {e}")

# Store the last customer message for each admin user
admin_last_customer_message = {}

# Track admin correction state
admin_correction_state = {}  # "waiting_for_answer" or "waiting_for_intent" or "selecting_intents"

# Track selected intents for multi-intent mode
admin_selected_intents = {}

# Keywords that trigger correction mode
CORRECTION_TRIGGERS = ["wrong", "wrong answer", "incorrect", "faux", "erreur", "pas correct"]

# --- ADMIN MENU TRIGGER WORDS ---
ADMIN_MENU_TRIGGERS = ["arimamy"]

# --- EXTEND ADMIN MENU BUTTONS ---
ADMIN_MENU_BUTTONS = [
    {"type": "postback", "title": "✅ Correct Intent", "payload": "CORRECT_INTENT_MENU"},
    {"type": "postback", "title": "📊 View Stats", "payload": "VIEW_STATS"},
    {"type": "postback", "title": "📋 View Live Corrections", "payload": "VIEW_LIVE_CORRECTIONS"},
    {"type": "postback", "title": "🗑️ Delete Live Correction", "payload": "DELETE_LIVE_CORRECTION"},
    {"type": "postback", "title": "❌ Cancel Correction Flow", "payload": "CANCEL_CORRECTION_FLOW"},
    {"type": "postback", "title": "🕓 Show Correction History", "payload": "SHOW_CORRECTION_HISTORY"},
    {"type": "postback", "title": "❓ Help", "payload": "ADMIN_HELP"}
]

# --- EXTEND STATE FOR ADMIN FLOWS ---
admin_pending_delete = {}  # sender_id: True if waiting for message to delete
admin_pending_history = {}  # sender_id: True if waiting for message to show history

def combine_responses(intents, lang="fr"):
    """Combine responses from multiple intents"""
    combined_parts = []
    
    for intent in intents:
        if intent in INTENT_RESPONSES and lang in INTENT_RESPONSES[intent]:
            combined_parts.append(INTENT_RESPONSES[intent][lang])
    
    if combined_parts:
        return "\n\n".join(combined_parts)
    else:
        return "Response not found for selected intents."

# Function to send a Messenger button template for admin correction options
async def send_correction_menu(recipient_id: str):
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    # Send buttons in groups of 3
    for i in range(0, len(ADMIN_MENU_BUTTONS), 3):
        buttons = ADMIN_MENU_BUTTONS[i:i+3]
        data = {
            "recipient": {"id": recipient_id},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "button",
                        "text": "What would you like to do?" if i == 0 else "More options:",
                        "buttons": buttons
                    }
                }
            }
        }
        try:
            response = requests.post(f"{GRAPH_API_URL}/me/messages", params=params, headers=headers, json=data)
            response.raise_for_status()
            logging.info(f"Correction menu sent successfully to {recipient_id}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending correction menu to {recipient_id}: {e}")
            if response is not None:
                logging.error(f"Response: {response.text}")

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Set up persistent menu on startup
    asyncio.run(setup_persistent_menu())
    
    uvicorn.run("app:app", host="0.0.0.0", port=8080)