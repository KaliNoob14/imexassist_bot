import os
import pickle
import logging
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths to model and embedding/label encoder files
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'intent_model.pt')
EMBEDDING_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'intent_embedding_model.txt')
LABEL_ENCODER_FILE = os.path.join(os.path.dirname(__file__), 'intent_label_encoder.pkl')

# Define the model architecture (must match training)
class IntentNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

intent_model = None
embedder = None
label_encoder = None

try:
    with open(EMBEDDING_MODEL_FILE, 'r') as f:
        embedding_model_name = f.read().strip()
    embedder = SentenceTransformer(embedding_model_name)
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    input_dim = embedder.get_sentence_embedding_dimension()
    num_classes = len(label_encoder.classes_)
    intent_model = IntentNet(input_dim, num_classes)
    intent_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    intent_model.eval()
    logging.info(f"Successfully loaded intent model from {MODEL_PATH}, embedder {embedding_model_name}, and label encoder from {LABEL_ENCODER_FILE}")
except Exception as e:
    logging.error(f"Failed to load intent model/embedder/label encoder: {e}", exc_info=True)

def predict_intent(text, threshold=0.5):
    """
    Predict the intent of the given text. Returns (intent, confidence).
    Returns (None, 0.0) if model/embedder/label_encoder is not loaded or confidence is too low.
    """
    if not intent_model or not embedder or not label_encoder:
        logging.error("Intent model, embedder, or label encoder not loaded.")
        return None, 0.0
    if not text or not text.strip():
        return None, 0.0
    try:
        X = embedder.encode([text])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = intent_model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        confidence = float(np.max(probs))
        intent_idx = int(np.argmax(probs))
        intent = label_encoder.inverse_transform([intent_idx])[0]
        if confidence < threshold:
            return None, confidence
        return intent, confidence
    except Exception as e:
        logging.error(f"Intent prediction failed: {e}", exc_info=True)
        return None, 0.0 