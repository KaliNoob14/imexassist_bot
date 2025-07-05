import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from dictionary.intent_keywords import INTENT_KEYWORDS

INPUT_FILE = "groupe_imex_mce_mbt_messages.json"
MODEL_FILE = "intent_model.pt"
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
    X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    # Build model
    model = IntentNet(X_train.shape[1], len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    print("Training model...")
    epochs = 10
    batch_size = 32
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        # Optionally print loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test).float().mean().item()
    print(f"Test accuracy: {acc*100:.2f}%")
    # Save model and vectorizer
    torch.save(model.state_dict(), MODEL_FILE)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump((vectorizer, label_encoder), f)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Vectorizer and label encoder saved to {VECTORIZER_FILE}")

if __name__ == "__main__":
    main() 