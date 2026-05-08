import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image

client = TestClient(app)

def test_predict_success():
    # สร้างรูปจำลองขนาด 224x224
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "predicted_class" in response.json()
    assert response.json()["message"] == "Success"

def test_predict_invalid_file():
    # ส่งไฟล์ text แทนรูปภาพ
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image content", "text/plain")}
    )
    assert response.status_code == 400