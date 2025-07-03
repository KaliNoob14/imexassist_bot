# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install build tools for fasttext and other native extensions
RUN apt-get update && apt-get install -y build-essential curl

# Download the fastText language identification model
RUN curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and the fastText model into the container
COPY . .

# Cloud Run expects the application to listen on the port specified by the PORT environment variable
ENV PORT 8080

# Command to run the application using Uvicorn
# It assumes your FastAPI app instance is named 'app' in 'app.py'
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

RUN ls -lh /app