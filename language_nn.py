import fasttext
import logging
from lingua import Language, LanguageDetectorBuilder

# Load the fastText model once at import time
MODEL_PATH = "lid.176.bin"
try:
    model = fasttext.load_model(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load fastText model from {MODEL_PATH}: {e}")
    model = None

# Lingua setup: only support English, French
LINGUA_LANGS = [Language.ENGLISH, Language.FRENCH]
lingua_detector = LanguageDetectorBuilder.from_languages(*LINGUA_LANGS).build()

DEFAULT_CONFIDENCE_THRESHOLD = 0.5

LANG_LABELS = {
    "__label__en": "en",
    "__label__fr": "fr",
    "__label__mg": "mg",
}

ISO_TO_LABEL = {
    "en": "__label__en",
    "fr": "__label__fr",
    "mg": "__label__mg",
}

def detect_language(text, threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Detect the language of the given text using fastText and Lingua.
    Returns (lang_code, combined_confidence).
    """
    if not text or not text.strip():
        return None, 0.0
    # Try fastText first
    ft_lang, ft_conf = None, 0.0
    if model:
        try:
            labels, confidences = model.predict(text)
            if labels and confidences:
                ft_label = labels[0]
                ft_conf = confidences[0]
                ft_lang = LANG_LABELS.get(ft_label, None)
        except Exception as e:
            logging.error(f"fastText detection failed: {e}", exc_info=True)
    # If fastText is confident enough, return
    if ft_lang and ft_conf >= threshold:
        return ft_lang, ft_conf
    # Fallback to Lingua
    try:
        lingua_confidences = lingua_detector.compute_language_confidence_values(text)
        lingua_scores = {c.language.iso_code_639_1.name.lower(): c.value for c in lingua_confidences}
        # Only consider supported languages
        lingua_scores = {k: v for k, v in lingua_scores.items() if k in ISO_TO_LABEL}
        if not lingua_scores:
            return None, 0.0
        # Combine fastText and Lingua probabilities if both available
        if ft_lang in lingua_scores:
            combined_score = (ft_conf + lingua_scores[ft_lang]) / 2
            return ft_lang, combined_score
        # Otherwise, pick the top Lingua result
        lingua_lang, lingua_conf = max(lingua_scores.items(), key=lambda x: x[1])
        return lingua_lang, lingua_conf
    except Exception as e:
        logging.error(f"Lingua detection failed: {e}", exc_info=True)
        return ft_lang, ft_conf if ft_lang else (None, 0.0) 