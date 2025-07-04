import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers
from dictionary.intent_keywords import INTENT_KEYWORDS

INPUT_FILE = "groupe_imex_mce_mbt_messages.json"
MODEL_FILE = "intent_model.h5"
VECTORIZER_FILE = "intent_vectorizer.pkl"

# Assign intent based on keywords
def assign_intent(text):
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return intent
    return None

def load_data():
    messages = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for msg in data:
            if msg.get('sender_type') != 'customer':
                continue
            text = msg.get('text', '').strip()
            intent = assign_intent(text)
            if intent:
                messages.append((text, intent))
    return messages

def main():
    print("Loading and labeling data...")
    data = load_data()
    if not data:
        print("No labeled data found.")
        return
    texts, intents = zip(*data)
    print("Sample count per intent:")
    for intent, count in Counter(intents).items():
        print(f"  {intent}: {count}")
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(intents)
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(texts)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Build model
    model = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Training model...")
    model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32, validation_split=0.1)
    # Evaluate
    loss, acc = model.evaluate(X_test.toarray(), y_test, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}%")
    # Save model and vectorizer
    model.save(MODEL_FILE)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump((vectorizer, label_encoder), f)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Vectorizer and label encoder saved to {VECTORIZER_FILE}")

if __name__ == "__main__":
    main() 