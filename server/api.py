import os
import time
import uuid
import shutil
import asyncio
import cv2
import numpy as np
import torch
import trimesh
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.preprocess import extract_silhouette_and_overlay, make_shaded_rgba
from pipeline.inflate import mesh_by_inflation, finalize_mesh
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


@app.get("/health")
@app.get("/ping")
def health_check():
    device, gpu_name = get_gpu_info()
    return {"status": "ok", "device": device, "gpu_name": gpu_name}


@app.post("/warmup")
async def warmup():
    device, gpu_name = get_gpu_info()
    loaded = []
    if device == "cuda":
        try:
            from pipeline.reconstruct_3d import load_triposr
            load_triposr()
            loaded.append("TripoSR")
        except Exception as e:
            loaded.append(f"TripoSR_err: {e}")
    return {"status": "warmed", "device": device, "gpu_name": gpu_name, "loaded": loaded}


@app.post("/generate")
async def generate(
    sketch: UploadFile = File(...),
    prompt: str = Form(""),
    mode: str = Form("ai"),          # "ai", "sketch", "inflate"
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
            # 1. Read sketch -> Silhouette + Overlay ("What the AI reads")
            gray, mask, strokes, overlay_bgr = extract_silhouette_and_overlay(input_path)
            overlay_path = f"{job_dir}/overlay.png"
            mask_path = f"{job_dir}/mask.png"
            cv2.imwrite(overlay_path, overlay_bgr)
            cv2.imwrite(mask_path, mask)

            rgba_for_3d = make_shaded_rgba(gray, mask, strokes)
            ai_img_url = None
            outline_url = f"/{mask_path}"
            refinement_iou = 1.0

            # 2. Optional AI Refinement stage (SD 1.5 + ControlNet Scribble + LCM-LoRA + rembg)
            device, _ = get_gpu_info()
            if mode == "ai" and device == "cuda":
                try:
                    from pipeline.refine_sd import refine_sketch_with_ai
                    ai_rgba, ai_outline, r_iou = refine_sketch_with_ai(strokes, mask, prompt, base_seed=seed)
                    refinement_iou = round(r_iou, 3)

                    ai_img_path = f"{job_dir}/ai_refined.png"
                    ai_outline_path = f"{job_dir}/ai_outline.png"
                    ai_rgba.save(ai_img_path)
                    cv2.imwrite(ai_outline_path, ai_outline)

                    ai_img_url = f"/{ai_img_path}"
                    outline_url = f"/{ai_outline_path}"

                    # Only use AI image if IoU >= 0.40; otherwise keep faithful sketch shaded RGBA
                    if r_iou >= 0.40:
                        rgba_for_3d = ai_rgba
                except Exception as e:
                    print(f"[WARN] AI stage fallback to sketch mode: {e}")

            # 3. 3D Reconstruction (TripoSR / SF3D on GPU, or 2.5D inflation fallback)
            t_mesh0 = time.time()
            backend_used = "2.5D Inflation (CPU)"
            if mode != "inflate":
                try:
                    from pipeline.reconstruct_3d import reconstruct_mesh
                    mesh, backend_used = reconstruct_mesh(rgba_for_3d, backend=model, resolution=resolution)
                except Exception as e:
                    print(f"[WARN] 3D model fallback to inflation ({e})")
                    mesh = mesh_by_inflation(mask, resolution=min(resolution, 160))
            else:
                mesh = mesh_by_inflation(mask, resolution=min(resolution, 160))
            mesh_time = round(time.time() - t_mesh0, 2)

            # 4. Export .glb and .obj + reload validation
            glb_path = f"{job_dir}/model.glb"
            obj_path = f"{job_dir}/model.obj"
            mesh.export(glb_path)
            mesh.export(obj_path)

            # Reload validation test
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


@app.post("/carve")
async def carve_endpoint(
    job_id: str = Form(...),
    side: str = Form("right"),
    sketch: UploadFile = File(...),
):
    async with GPU_LOCK:
        job_dir = f"outputs/{job_id}"
        glb_path = f"{job_dir}/model.glb"
        mask_path = f"{job_dir}/mask.png"
        if not os.path.exists(glb_path):
            return JSONResponse({"error": "Job ID not found"}, status_code=404)

        side_input = f"{job_dir}/side_input.png"
        with open(side_input, "wb") as buffer:
            shutil.copyfileobj(sketch.file, buffer)

        _, side_mask, _, _ = extract_silhouette_and_overlay(side_input)
        mesh = trimesh.load(glb_path, force="mesh")
        carved = carve_mesh_with_side_silhouette(mesh, side_mask, view_side=side)

        carved_glb = f"{job_dir}/model_carved.glb"
        carved_obj = f"{job_dir}/model_carved.obj"
        carved.export(carved_glb)
        carved.export(carved_obj)

        orig_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        montage_path = f"{job_dir}/views_carved.png"
        front_iou = render_six_views_and_iou(carved, orig_mask, montage_path)

        return JSONResponse({
            "job_id": job_id,
            "glb_url": f"/{carved_glb}?t={int(time.time())}",
            "obj_url": f"/{carved_obj}?t={int(time.time())}",
            "montage_url": f"/{montage_path}?t={int(time.time())}",
            "front_iou": round(front_iou, 3),
            "vertices": f"{len(carved.vertices):,}",
            "faces": f"{len(carved.faces):,}",
        })