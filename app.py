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
from modules.admin_correction import (
    admin_correction_state, admin_last_customer_message, admin_selected_intents, CORRECTION_TRIGGERS,
    strip_emojis, normalize_message, load_live_corrections, save_live_correction, parse_correction, apply_correction,
    send_correction_menu, send_correction_mode_menu, send_intent_selection_menu,
    view_live_corrections, delete_live_correction, cancel_correction_flow, show_correction_history
)
from fastapi.responses import PlainTextResponse

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

@app.get("/webhook")
def verify_webhook(request: Request):
    """Facebook webhook verification endpoint (GET)"""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="Verification failed", status_code=403)

@app.post("/webhook")
async def handle_webhook(request: Request):
    """Facebook webhook event handler (POST)"""
    body = await request.body()
    logging.info(f"Received webhook event: {body}")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Set up persistent menu on startup
    asyncio.run(send_correction_menu(ADMIN_SENDER_IDS[0])) # Example: send to the first admin ID
    
    uvicorn.run("app:app", host="0.0.0.0", port=8080)