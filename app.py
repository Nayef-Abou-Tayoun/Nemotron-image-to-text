from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

app = FastAPI()

# Load model at startup (CPU-optimized)
print("Loading Nemotron OCR v2 model...")
processor = AutoProcessor.from_pretrained("nvidia/nemotron-ocr-v2", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "nvidia/nemotron-ocr-v2",
    trust_remote_code=True,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu"
)
print("Model loaded successfully!")

@app.get("/")
def health():
    return {"status": "ok", "model": "nemotron-ocr-v2", "device": "cpu"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    OCR endpoint that accepts an image file and returns extracted text.
    Optimized for CPU inference.
    """
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run OCR inference
        inputs = processor(images=image, return_tensors="pt")
        
        # Move inputs to CPU explicitly
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Generate OCR output
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        
        # Decode the output
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
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
