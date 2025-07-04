import os
import pickle
import logging
import keras

# Paths to model and vectorizer files
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'intent_model.h5')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'intent_vectorizer.pkl')

# Load model and vectorizer at import time
intent_model = None
vectorizer = None
label_encoder = None

try:
    intent_model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load intent model from {MODEL_PATH}: {e}")

try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer, label_encoder = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load vectorizer/label encoder from {VECTORIZER_PATH}: {e}")

def predict_intent(text, threshold=0.5):
    """
    Predict the intent of the given text. Returns (intent, confidence).
    Returns (None, 0.0) if model/vectorizer/label_encoder is not loaded or confidence is too low.
    """
    if not intent_model or not vectorizer or not label_encoder:
        logging.error("Intent model, vectorizer, or label encoder not loaded.")
        return None, 0.0
    if not text or not text.strip():
        return None, 0.0
    try:
        X = vectorizer.transform([text])
        probs = intent_model.predict(X.toarray())
        confidence = float(probs.max())
        intent_idx = int(probs.argmax())
        intent = label_encoder.inverse_transform([intent_idx])[0]
        if confidence < threshold:
            return None, confidence
        return intent, confidence
    except Exception as e:
        logging.error(f"Intent prediction failed: {e}", exc_info=True)
        return None, 0.0 