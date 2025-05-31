FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY chatterbox_openai_api.py .
COPY . .

# Install chatterbox if it's a local package
RUN pip install -e .

EXPOSE 8000

CMD ["python", "chatterbox_openai_api.py", "--host", "0.0.0.0", "--port", "8000"]