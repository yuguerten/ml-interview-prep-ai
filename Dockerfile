# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV HUGGINGFACE_API_TOKEN=$HUGGINGFACE_API_TOKEN

# Run the application
CMD ["streamlit", "run", "app.py"]