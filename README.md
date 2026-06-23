# Nemotron OCR v2 - FastAPI Service

FastAPI-based OCR service using NVIDIA's Nemotron OCR v2 model, optimized for CPU deployment on IBM Code Engine.

## Project Structure

```
nemotron-ocr-ce/
├── app.py              # FastAPI application with OCR endpoint
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
└── README.md          # This file
```

## Local Development

### 1. Test with Docker

Build the container:
```bash
docker build -t nemotron-ocr-ce .
```

Run locally:
```bash
docker run -p 8080:8080 nemotron-ocr-ce
```

Test the health endpoint:
```bash
curl http://localhost:8080/
```

Test OCR with an image:
```bash
curl -X POST "http://localhost:8080/ocr" \
  -F "file=@your-image.png"
```

## Deploy to IBM Code Engine

### 1. Login to IBM Cloud

```bash
ibmcloud login --sso
ibmcloud target -r us-south
ibmcloud plugin install code-engine
```

### 2. Create Code Engine Project

```bash
ibmcloud ce project create --name nemotron-ocr-project
ibmcloud ce project select --name nemotron-ocr-project
```

### 3. Deploy Application

Deploy from local source (CPU-optimized):
```bash
ibmcloud ce application create \
  --name nemotron-ocr \
  --build-source . \
  --cpu 4 \
  --memory 16G \
  --min-scale 0 \
  --max-scale 1 \
  --port 8080
```

### 4. Get Application URL

```bash
ibmcloud ce application get --name nemotron-ocr --output url
```

### 5. Test Production Deployment

```bash
APP_URL=$(ibmcloud ce application get --name nemotron-ocr --output url)

curl -X POST "$APP_URL/ocr" \
  -F "file=@your-image.png"
```

## API Endpoints

### GET /
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "nemotron-ocr-v2",
  "device": "cpu"
}
```

### POST /ocr
OCR endpoint that accepts an image file and returns extracted text.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response (Success):**
```json
{
  "filename": "image.png",
  "text": "Extracted text from the image...",
  "width": 1920,
  "height": 1080,
  "status": "success"
}
```

**Response (Error):**
```json
{
  "filename": "image.png",
  "error": "Error message",
  "status": "error"
}
```

## Configuration

### CPU Optimization
- Model runs on CPU with `torch.float32` precision
- Configured for single image processing
- `--min-scale 0` enables scale-to-zero for cost savings
- First request after idle will have cold start delay (~30-60 seconds)

### Resource Allocation
- **CPU:** 4 vCPUs
- **Memory:** 16GB (required for model loading)
- **Scaling:** 0-1 instances (serverless)

## Notes

- **Latency:** CPU inference is slower than GPU but acceptable for single image processing
- **Cold Start:** First request after idle period will take longer due to model loading
- **Cost:** Only pay when processing requests (scales to zero when idle)
- **Model:** Uses NVIDIA Nemotron OCR v2 from Hugging Face

## Troubleshooting

### Model Loading Issues
If the model fails to load, ensure:
- Sufficient memory allocation (16GB minimum)
- Hugging Face Hub is accessible
- `trust_remote_code=True` is set in model loading

### Timeout Issues
For large images or slow CPU inference:
- Increase Code Engine timeout settings
- Consider resizing images before processing
- Monitor memory usage

## References

- [Nemotron OCR v2 on Hugging Face](https://huggingface.co/nvidia/nemotron-ocr-v2)
- [IBM Code Engine Documentation](https://cloud.ibm.com/docs/codeengine)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)