import os
import time
import uuid
import shutil
import asyncio
import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.preprocess import extract_silhouette_and_overlay, make_shaded_rgba
from pipeline.inflate import mesh_by_inflation
from pipeline.views import render_six_views_and_iou, carve_mesh_with_side_silhouette

app = FastAPI(title="Sketch2Shape AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

GPU_LOCK = asyncio.Lock()


def get_gpu_info():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple MPS"
    return "cpu", "CPU"


def make_color_rgba(input_path: str, mask: np.ndarray) -> Image.Image:
    """Preserves original RGB colors for uploaded photos/shaded drawings at 512x512."""
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    scale = 440.0 / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (512, 512), (255, 255, 255))
    x0, y0 = (512 - nw) // 2, (512 - nh) // 2
    canvas.paste(resized, (x0, y0))

    rgb_arr = np.array(canvas)
    rgba_arr = np.dstack([rgb_arr, mask])
    return Image.fromarray(rgba_arr, mode="RGBA")


@app.get("/health")
@app.get("/ping")
def health_check():
    device, gpu_name = get_gpu_info()
    return {"status": "ok", "device": device, "gpu_name": gpu_name}


@app.post("/warmup")
async def warmup():
    device, gpu_name = get_gpu_info()
    from pipeline.reconstruct_3d import load_triposr
    load_triposr()
    return {"status": "warmed", "device": device, "gpu_name": gpu_name, "loaded": ["TripoSR"]}


@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...),
    prompt: str = Form(""),
    mode: str = Form("sketch"),      # "sketch", "color", "ai", "inflate"
    model: str = Form("triposr"),    # "triposr" or "sf3d"
    resolution: int = Form(192),     # 128, 192, 256
    seed: int = Form(42),
):
    async with GPU_LOCK:
        t_start = time.time()
        job_id = str(uuid.uuid4())[:8]
        job_dir = f"outputs/{job_id}"
        os.makedirs(job_dir, exist_ok=True)

        input_path = f"{job_dir}/input.png"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(sketch.file, buffer)

        try:
            # 1. Extract silhouette & visual overlay
            gray, mask, strokes, overlay_bgr = extract_silhouette_and_overlay(input_path)
            overlay_path = f"{job_dir}/overlay.png"
            mask_path = f"{job_dir}/mask.png"
            cv2.imwrite(overlay_path, overlay_bgr)
            cv2.imwrite(mask_path, mask)

            # Prepare RGBA input based on mode
            if mode == "color":
                rgba_for_3d = make_color_rgba(input_path, mask)
            else:
                rgba_for_3d = make_shaded_rgba(gray, mask, strokes)

            ai_img_url = None
            outline_url = f"/{mask_path}"
            refinement_iou = 1.0

            # 2. Optional AI Refinement stage (SD 1.5 + ControlNet Scribble + LCM-LoRA + rembg)
            device, _ = get_gpu_info()
            if mode == "ai" and device == "cuda":
                from pipeline.refine_sd import refine_sketch_with_ai
                ai_rgba, ai_outline, r_iou = refine_sketch_with_ai(strokes, mask, prompt, base_seed=seed)
                refinement_iou = round(r_iou, 3)

                ai_img_path = f"{job_dir}/ai_refined.png"
                ai_outline_path = f"{job_dir}/ai_outline.png"
                ai_rgba.save(ai_img_path)
                cv2.imwrite(ai_outline_path, ai_outline)

                ai_img_url = f"/{ai_img_path}"
                outline_url = f"/{ai_outline_path}"
                if r_iou >= 0.35:
                    rgba_for_3d = ai_rgba

            # 3. 3D Reconstruction on GPU (TripoSR / SF3D)
            t_mesh0 = time.time()
            if mode == "inflate":
                mesh = mesh_by_inflation(mask, resolution=min(resolution, 160))
                backend_used = "2.5D Inflation (CPU)"
            else:
                from pipeline.reconstruct_3d import reconstruct_mesh
                mesh, backend_used = reconstruct_mesh(rgba_for_3d, backend=model, resolution=resolution)
            mesh_time = round(time.time() - t_mesh0, 2)

            # 4. Export .glb and .obj + reload validation
            glb_path = f"{job_dir}/model.glb"
            obj_path = f"{job_dir}/model.obj"
            mesh.export(glb_path)
            mesh.export(obj_path)

            reloaded = trimesh.load(glb_path, force="mesh")
            extents = np.round(reloaded.extents, 2).tolist() if len(reloaded.vertices) else [0, 0, 0]

            # 5. Six Orthographic Silhouette Views + Front IoU
            montage_path = f"{job_dir}/views.png"
            front_iou = render_six_views_and_iou(reloaded, mask, montage_path)

            total_time = round(time.time() - t_start, 2)
            glb_kb = round(os.path.getsize(glb_path) / 1024, 1)
            obj_kb = round(os.path.getsize(obj_path) / 1024, 1)

            return JSONResponse({
                "job_id": job_id,
                "glb_url": f"/{glb_path}",
                "obj_url": f"/{obj_path}",
                "overlay_url": f"/{overlay_path}",
                "mask_url": f"/{mask_path}",
                "outline_url": outline_url,
                "ai_img_url": ai_img_url,
                "montage_url": f"/{montage_path}",
                "stats": {
                    "total_time": f"{total_time}s",
                    "mesh_time": f"{mesh_time}s",
                    "vertices": f"{len(reloaded.vertices):,}",
                    "faces": f"{len(reloaded.faces):,}",
                    "bbox": f"{extents[0]} × {extents[1]} × {extents[2]}",
                    "watertight": "Yes" if reloaded.is_watertight else "No",
                    "glb_size": f"{glb_kb} KB",
                    "obj_size": f"{obj_kb} KB",
                    "front_iou": round(front_iou, 3),
                    "refinement_iou": refinement_iou,
                    "backend_used": backend_used,
                }
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)