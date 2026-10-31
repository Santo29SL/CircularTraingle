from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
import cv2

from pipeline.preprocess import extract_silhouette
from pipeline.inflate import mesh_by_inflation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/health")
def health_check():
    return {"status": "ok", "device": os.environ.get("S2S_DEVICE", "cpu")}

@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...), 
    prompt: str = Form(""), 
    mode: str = Form("ai"),
    model: str = Form("sf3d") 
):
    job_id = str(uuid.uuid4())
    job_dir = f"outputs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    
    input_path = f"{job_dir}/input.png"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(sketch.file, buffer)
        
    try:
        mask = extract_silhouette(input_path)
        outline_path = f"{job_dir}/outline.png"
        cv2.imwrite(outline_path, mask)
        
        # GPU Routing Logic
        if model in ["sf3d", "triposr"] and os.environ.get("S2S_DEVICE") != "cpu":
            # NOTE: Implement actual SF3D/TripoSR inference here in Kaggle Phase
            pass 
            
        # Fallback to local CPU inflation if GPU is missing or while developing
        mesh = mesh_by_inflation(mask)
        glb_path = f"{job_dir}/model.glb"
        mesh.export(glb_path)
        
        return JSONResponse({
            "job_id": job_id,
            "glb_url": f"/{glb_path}",
            "outline_url": f"/{outline_path}",
            "model_used": model
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)