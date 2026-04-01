import os
import base64
import io
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

from inference_sdk import InferenceHTTPClient, InferenceConfiguration

app = FastAPI(
    title="Gender Detection API",
    description="API for gender detection using Roboflow Inference SDK",
    version="1.0.0"
)

ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", "teguh-rijanandi")
WORKFLOW_ID = os.getenv("WORKFLOW_ID", "find-females-and-males")

client: Optional[InferenceHTTPClient] = None


class PredictionResult(BaseModel):
    gender: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


class PredictionResponse(BaseModel):
    success: bool
    predictions: List[PredictionResult]
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    message: Optional[str] = None


def initialize_client():
    global client
    if not ROBOFLOW_API_KEY:
        raise ValueError("ROBOFLOW_API_KEY environment variable is not set")
    
    client = InferenceHTTPClient.init(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )
    return client


@app.on_event("startup")
async def startup_event():
    initialize_client()


@app.get("/")
async def root():
    return {
        "message": "Gender Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "roboflow_config": {
            "workspace": WORKSPACE_NAME,
            "workflow_id": WORKFLOW_ID
        }
    }


def process_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


async def save_upload_file(upload_file: UploadFile, path: str) -> None:
    content = await upload_file.read()
    with open(path, "wb") as f:
        f.write(content)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...)
):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        
        temp_image_path = "/tmp/input_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(contents)
        
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_image_path}
        )
        
        predictions = []
        image_width = None
        image_height = None
        
        if isinstance(result, list) and len(result) > 0:
            pred_data = result[0].get('predictions', {}).get('predictions', [])
            
            for pred in pred_data:
                predictions.append(PredictionResult(
                    gender=pred.get('class', 'Unknown'),
                    confidence=float(pred.get('confidence', 0.0)),
                    x=float(pred.get('x', 0)),
                    y=float(pred.get('y', 0)),
                    width=float(pred.get('width', 0)),
                    height=float(pred.get('height', 0))
                ))
        
        try:
            img = Image.open(io.BytesIO(contents))
            image_width, image_height = img.size
        except:
            pass
        
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return PredictionResponse(
            success=True,
            predictions=predictions,
            image_width=image_width,
            image_height=image_height,
            message=f"Found {len(predictions)} prediction(s)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(image_data: str = Form(...)):
    if not image_data:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        
        temp_image_path = "/tmp/input_image_base64.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_image_path}
        )
        
        predictions = []
        image_width = None
        image_height = None
        
        if isinstance(result, list) and len(result) > 0:
            pred_data = result[0].get('predictions', {}).get('predictions', [])
            
            for pred in pred_data:
                predictions.append(PredictionResult(
                    gender=pred.get('class', 'Unknown'),
                    confidence=float(pred.get('confidence', 0.0)),
                    x=float(pred.get('x', 0)),
                    y=float(pred.get('y', 0)),
                    width=float(pred.get('width', 0)),
                    height=float(pred.get('height', 0))
                ))
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            image_width, image_height = img.size
        except:
            pass
        
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return PredictionResponse(
            success=True,
            predictions=predictions,
            image_width=image_width,
            image_height=image_height,
            message=f"Found {len(predictions)} prediction(s)"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
