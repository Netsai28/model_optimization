import io
import asyncio
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor

# ตั้งค่า Path ให้หาไฟล์โมเดลเจอ (อยู่ที่โฟลเดอร์นอก app)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "hf_resnet50_int8.onnx")

app = FastAPI(title="Image Classification API")

class PredictionResponse(BaseModel):
    predicted_class: int
    message: str

def run_inference(image_bytes: bytes) -> int:
    try:
        # โหลดโมเดลภายใน worker process
        session = ort.InferenceSession(MODEL_PATH)
        
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        
        # [สำคัญ] บังคับเป็น float32 ตามที่ ONNX ต้องการ
        img_data = np.array(img).astype(np.float32).transpose(2, 0, 1) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_data = (img_data - mean) / std
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
        
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: img_data})
        
        return int(np.argmax(result[0]))
    except Exception as e:
        raise RuntimeError(f"Inference Error: {str(e)}")

# โจทย์ข้อ 2: Concurrency ด้วย ProcessPoolExecutor
executor = ProcessPoolExecutor(max_workers=2)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # โจทย์ข้อ 3: Production Error Handling
    contents = await file.read()
    
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (Max 5MB)")
        
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    loop = asyncio.get_running_loop()
    try:
        predicted_class = await loop.run_in_executor(executor, run_inference, contents)
        return PredictionResponse(predicted_class=predicted_class, message="Success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))