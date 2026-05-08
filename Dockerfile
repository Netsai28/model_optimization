FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง libgomp1 สำหรับ onnxruntime (อิงตามสไลด์อาจารย์)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port 7860 สำหรับ Hugging Face Spaces
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]