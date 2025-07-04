# app.py (Simplified for quick test)
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv
import os
import requests
import logging
import time
from langdetect import detect, LangDetectException
from language_nn import detect_language
from intent_nn import predict_intent

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
    print(f"[DEBUG] /webhook POST called")
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

                    # --- Check for empty or whitespace message ---
                    if not message_text or not message_text.strip():
                        print(f"[DEBUG] Empty or whitespace message received from {sender_id}")
                        reply = "Sorry, I couldn't detect your language."
                        await send_text_message(sender_id, reply)
                        continue

                    # --- Rate limiting check ---
                    if is_rate_limited(sender_id, message_text):
                        print(f"[DEBUG] Rate limited message from {sender_id}: {message_text}")
                        continue

                    # --- fastText language detection and response ---
                    lang_code, confidence = detect_language(message_text)
                    print(f"[DEBUG] fastText detected: {lang_code} (confidence: {confidence:.2f})")

                    # --- Intent detection ---
                    intent, intent_conf = predict_intent(message_text)
                    print(f"[DEBUG] Predicted intent: {intent} (confidence: {intent_conf:.2f})")

                    # --- Intent-based response logic ---
                    reply_lang = lang_code if lang_code in ["fr", "mg", "en"] else "en"
                    if intent and intent_conf >= 0.5 and intent in INTENT_RESPONSES:
                        reply = INTENT_RESPONSES[intent][reply_lang]
                    else:
                        reply = DEFAULT_RESPONSES[reply_lang]

                    await send_text_message(sender_id, reply)
                    # --- End intent-based response logic ---

                # Handle postback from buttons/persistent menu
                elif "postback" in messaging_event:
                    payload = messaging_event["postback"]["payload"]
                    print(f"[DEBUG] Received postback from {sender_id}: {payload}")
                    await send_text_message(sender_id, f"Received payload: {payload}")
    return Response(content="OK", status_code=200)

# You can add a simple root endpoint for health checks or general info
@app.get("/")
async def root():
    logging.info("Root endpoint hit!")
    return {"message": "Hello, I'm IMEX Assist, how may I help you?"}

# Intent to response mapping
INTENT_RESPONSES = {
    "pricing": {
        "fr": "Nos tarifs varient selon le service. Pourriez-vous préciser votre demande?",
        "mg": "Miovaova arakaraka ny tolotra ny vidiny. Azafady hazavao ny fangatahanao.",
        "en": "Our prices vary by service. Could you specify your request?"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)