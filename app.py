# app.py (Simplified for quick test)
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv
import os
import requests
import logging

load_dotenv() # Load environment variables from .env during local dev

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Retrieve these from environment variables set during Cloud Run deployment
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

GRAPH_API_URL = "https://graph.facebook.com/v19.0" # Use a recent API version

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
    logging.info(f"[DEBUG] /webhook POST called")
    logging.info(f"[DEBUG] Received webhook data: {data}")

    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                sender_id = messaging_event["sender"]["id"]

                # Handle text messages
                if "message" in messaging_event and "text" in messaging_event["message"]:
                    message_text = messaging_event["message"]["text"]
                    logging.info(f"[DEBUG] Received message from {sender_id}: {message_text}")
                    await send_text_message(sender_id, f"Received: {message_text} - Bot is alive!")

                # Handle postback from buttons/persistent menu
                elif "postback" in messaging_event:
                    payload = messaging_event["postback"]["payload"]
                    logging.info(f"[DEBUG] Received postback from {sender_id}: {payload}")
                    await send_text_message(sender_id, f"Received payload: {payload}")
    return Response(content="OK", status_code=200)

# You can add a simple root endpoint for health checks or general info
@app.get("/")
async def root():
    logging.info("Root endpoint hit!")
    return {"message": "Hello, I'm IMEX Assist, how may I help you today?"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)