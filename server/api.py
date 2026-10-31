from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "device": os.environ.get("S2S_DEVICE", "cpu")}

@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...), 
    prompt: str = Form(""), 
    mode: str = Form("ai")
):
    job_id = str(uuid.uuid4())
    return JSONResponse({
        "job_id": job_id,
        "glb_url": f"/outputs/{job_id}/model.glb",
        "montage_url": f"/outputs/{job_id}/views.png"
    })