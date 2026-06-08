FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Defaults to paper mode against the OANDA practice (demo) endpoint
# configured in config/settings.yaml. Educational/research use only.
CMD ["python", "main.py", "paper"]
