# Use an official Python 3.10+ image as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install build tools and dependencies for numpy/scipy
RUN apt-get update && apt-get install -y build-essential curl gfortran libopenblas-dev liblapack-dev

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download Hugging Face model and tokenizer for offline use
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('papluca/xlm-roberta-base-language-detection'); \
    AutoModelForSequenceClassification.from_pretrained('papluca/xlm-roberta-base-language-detection')"

# Pre-download sentence-transformers model for intent detection
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of your application code into the container
COPY . .

# Copy the intent model, embedding model name, and label encoder into the container
COPY intent_model.pt intent_embedding_model.txt intent_label_encoder.pkl /app/

# Cloud Run expects the application to listen on the port specified by the PORT environment variable
ENV PORT 8080

# Set Hugging Face Transformers to offline mode
ENV TRANSFORMERS_OFFLINE=1

# Command to run the application using Uvicorn
# It assumes your FastAPI app instance is named 'app' in 'app.py'
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

RUN ls -lh /app