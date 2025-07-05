import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from lingua import Language, LanguageDetectorBuilder

# Load the transformer-based language detection model
MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
except Exception as e:
    logging.error(f"Failed to load transformer language detection model: {e}")
    tokenizer = None
    model = None

SUPPORTED_LANGS = {"en", "fr"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

LABEL_TO_LANG = {
    0: "af", 1: "am", 2: "ar", 3: "as", 4: "az", 5: "be", 6: "bg", 7: "bn", 8: "bo", 9: "bs", 10: "ca", 11: "cs", 12: "cy", 13: "da", 14: "de", 15: "dv", 16: "el", 17: "en", 18: "es", 19: "et", 20: "eu", 21: "fa", 22: "fi", 23: "fo", 24: "fr", 25: "ga", 26: "gl", 27: "gu", 28: "he", 29: "hi", 30: "hr", 31: "hu", 32: "hy", 33: "id", 34: "is", 35: "it", 36: "ja", 37: "jv", 38: "ka", 39: "kk", 40: "km", 41: "kn", 42: "ko", 43: "ku", 44: "ky", 45: "la", 46: "lb", 47: "lo", 48: "lt", 49: "lv", 50: "mg", 51: "mk", 52: "ml", 53: "mn", 54: "mr", 55: "ms", 56: "mt", 57: "ne", 58: "nl", 59: "no", 60: "oc", 61: "or", 62: "pa", 63: "pl", 64: "ps", 65: "pt", 66: "qu", 67: "rm", 68: "ro", 69: "ru", 70: "rw", 71: "se", 72: "si", 73: "sk", 74: "sl", 75: "so", 76: "sq", 77: "sr", 78: "sv", 79: "sw", 80: "ta", 81: "te", 82: "tg", 83: "th", 84: "tk", 85: "tl", 86: "tn", 87: "tr", 88: "tt", 89: "ug", 90: "uk", 91: "ur", 92: "uz", 93: "vi", 94: "vo", 95: "wa", 96: "xh", 97: "yi", 98: "yo", 99: "zh"
}

# Lingua setup: only support English, French
LINGUA_LANGS = [Language.ENGLISH, Language.FRENCH]
lingua_detector = LanguageDetectorBuilder.from_languages(*LINGUA_LANGS).build()

def detect_language(text, threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Detect the language of the given text using a transformer model, fallback to Lingua.
    Returns (lang_code, confidence).
    Only supports English and French for now.
    """
    if not text or not text.strip():
        return None, 0.0
    # Try transformer model first
    if model and tokenizer:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                top_idx = int(torch.argmax(probs))
                lang = LABEL_TO_LANG.get(top_idx, None)
                confidence = float(probs[top_idx])
                if lang in SUPPORTED_LANGS and confidence >= threshold:
                    return lang, confidence
        except Exception as e:
            logging.error(f"Transformer language detection failed: {e}", exc_info=True)
    # Fallback to Lingua
    try:
        lingua_lang = lingua_detector.detect_language_of(text)
        if lingua_lang:
            lingua_code = lingua_lang.iso_code_639_1.name.lower()
            if lingua_code in SUPPORTED_LANGS:
                # Lingua does not provide a confidence, so use 0.5 as a default
                return lingua_code, 0.5
    except Exception as e:
        logging.error(f"Lingua detection failed: {e}", exc_info=True)
    return None, 0.0 