from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

app = FastAPI()

# Load model and processor at startup (CPU-optimized)
print("Loading Nemotron OCR v2 model...")
try:
    model = AutoModelForVision2Seq.from_pretrained(
        "nvidia/nemotron-ocr-v2",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(
        "nvidia/nemotron-ocr-v2",
        trust_remote_code=True
    )
    print("Model and processor loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    processor = None

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
        
        # Process image with processor
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate text with model
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        # Decode the generated text
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
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
