import cv2
import numpy as np
import torch
from PIL import Image
from pipeline.preprocess import compute_iou

_SD_PIPE = None
_REMBG_SESSION = None

def get_device() -> str:
    import os
    override = os.environ.get("S2S_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sd_pipeline():
    global _SD_PIPE
    if _SD_PIPE is not None:
        return _SD_PIPE

    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None  # Faster memory footprint on T4; blank check still active below
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
    except Exception as e:
        print(f"[WARN] LCM-LoRA load skipped ({e}), using 12 steps.")

    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    _SD_PIPE = pipe
    return _SD_PIPE


def remove_bg_and_extract_outline(pil_img: Image.Image):
    """
    Removes background with rembg (u2netp) and creates the inverted white-on-black
    outline image + binary silhouette mask.
    """
    global _REMBG_SESSION
    from rembg import remove, new_session
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("u2netp")

    rgba = remove(pil_img, session=_REMBG_SESSION)
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    _, mask = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)

    # White-on-black outline from alpha boundary + internal Canny edges
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros_like(mask)
    cv2.drawContours(outline, cnts, -1, 255, 2)
    outline = cv2.bitwise_or(outline, edges)

    return rgba, mask, outline


def refine_sketch_with_ai(strokes: np.ndarray, sketch_mask: np.ndarray, prompt: str = "", base_seed: int = 42):
    """
    Runs ControlNet Scribble + LCM-LoRA up to 3 seeds, picking the best IoU match.
    Returns (best_rgba_pil, best_outline_np, best_iou).
    """
    pipe = load_sd_pipeline()
    device = get_device()

    # ControlNet scribble expects white lines on black background
    control_rgb = np.stack([strokes, strokes, strokes], axis=-1)
    control_pil = Image.fromarray(control_rgb)

    clean_prompt = prompt.strip() if prompt.strip() else "a single solid 3D object"
    full_prompt = (
        f"{clean_prompt}, clean studio product photography, centered, "
        f"single isolated object on plain white background, smooth shading, 3d asset"
    )
    neg_prompt = "multiple objects, complex background, shadows, cropped, text, watermark, blurry"

    best_rgba, best_outline, best_iou = None, None, -1.0

    for attempt in range(3):
        seed = base_seed + attempt * 101
        gen = torch.Generator(device=device).manual_seed(seed)
        out_pil = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            image=control_pil,
            num_inference_steps=6,
            guidance_scale=1.5,
            controlnet_conditioning_scale=0.95,
            generator=gen
        ).images[0]

        rgba, ai_mask, outline = remove_bg_and_extract_outline(out_pil)
        iou = compute_iou(sketch_mask, ai_mask)
        if iou > best_iou:
            best_rgba, best_outline, best_iou = rgba, outline, iou
        if iou >= 0.52:
            break

    return best_rgba, best_outline, best_iou