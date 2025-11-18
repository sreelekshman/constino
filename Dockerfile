FROM python:3.10-slim

# Set environment variables for cache and huggingface
ENV HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    SENTENCE_TRANSFORMERS_HOME=/app/cache

# Set the working directory in the container
WORKDIR /app

# Install basic OS dependencies (curl for debugging, git and build tools if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make sure the cache directory exists and is writable
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Expose the port the app runs on (if applicable, e.g., Gradio/Flask)
EXPOSE 7860

# Default command to run the app
CMD ["python", "app.py"]