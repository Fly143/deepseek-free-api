FROM python:3.10-slim

RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN test -f config.json || cp config.example.json config.json

EXPOSE 8000

CMD ["python", "proxy.py"]
