# Base image Python
FROM python:3.11-slim-bookworm


RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless procps build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set Java Home
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Set working directory
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode ke dalam container
COPY . .

# Default command (akan ditimpa oleh docker-compose)
CMD ["python3"]