# Gender Detection API

FastAPI application for gender detection using Roboflow Inference SDK.

## Project Structure

```
.
├── main.py           # FastAPI application with /predict endpoint
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker container configuration
├── .env.example      # Environment variables template
└── README.md         # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your Roboflow API key
```

## Running

### Local Development
```bash
uvicorn main:app --reload --port 8000
```

### Docker

Build:
```bash
docker build -t gender-detection-api .
```

Run:
```bash
docker run -p 8000:8000 \
  -e ROBOFLOW_API_KEY=your_api_key \
  gender-detection-api
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Upload image for prediction
- `POST /predict/base64` - Submit base64 encoded image

## API Usage

### POST /predict

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

Response:
```json
{
  "success": true,
  "predictions": [
    {
      "gender": "Male",
      "confidence": 0.95,
      "x": 320,
      "y": 240,
      "width": 100,
      "height": 150
    }
  ],
  "image_width": 640,
  "image_height": 480,
  "message": "Found 1 prediction(s)"
}
```
# face-detection
