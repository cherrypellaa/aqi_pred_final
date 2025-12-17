# Base image Python
FROM python:3.11-slim-bookwor

RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless procps build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x run_all.sh

CMD ["./run_all.sh"]