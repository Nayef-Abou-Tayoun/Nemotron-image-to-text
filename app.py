from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from transformers import AutoModel
import torch

app = FastAPI()

# Load model at startup (CPU-optimized)
print("Loading Nemotron OCR v2 model...")
try:
    model = AutoModel.from_pretrained(
        "nvidia/nemotron-ocr-v2",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu"
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model": "nemotron-ocr-v2",
        "device": "cpu",
        "model_loaded": model is not None
    }

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    OCR endpoint that accepts an image file and returns extracted text.
    Optimized for CPU inference.
    """
    if model is None:
        return {
            "filename": file.filename,
            "error": "Model not loaded",
            "status": "error"
        }
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # The model has its own processing built-in with trust_remote_code
        # Call the model directly with the image
        with torch.no_grad():
            result = model.generate(image, max_new_tokens=512)
        
        # Extract text from result
        if isinstance(result, str):
            text = result
        elif hasattr(result, 'text'):
            text = result.text
        else:
            text = str(result)
        
        return {
            "filename": file.filename,
            "text": text,
            "width": image.width,
            "height": image.height,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "filename": file.filename,
            "error": str(e),
            "status": "error"
        }

# Made with Bob
