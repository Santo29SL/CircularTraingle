import cv2
import numpy as np
from PIL import Image

def extract_silhouette_and_overlay(image_path: str):
    """
    Returns:
      - gray: normalized grayscale sketch (white background, dark strokes)
      - mask: binary uint8 silhouette (255 = object, 0 = background)
      - strokes: binary uint8 strokes (255 = stroke, 0 = background)
      - overlay_bgr: visual 'What the AI reads' panel (dark bg, soft fill, bright strokes)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Handle RGBA transparency if uploaded PNG has alpha
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        comp = (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
        gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize to 512x512 while preserving aspect ratio with padding
    h, w = gray.shape[:2]
    scale = 440.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    # Determine if dark background or white background using border pixels
    border = np.concatenate([resized[0, :], resized[-1, :], resized[:, 0], resized[:, -1]])
    bg_val = 255 if np.median(border) >= 127 else 0
    canvas = np.full((512, 512), bg_val, dtype=np.uint8)
    y0, x0 = (512 - nh) // 2, (512 - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    # Invert if dark background so strokes are dark on white (255)
    if bg_val == 0:
        canvas = 255 - canvas

    # Extract strokes (255 where ink is present)
    _, strokes = cv2.threshold(255 - canvas, 28, 255, cv2.THRESH_BINARY)

    # Close small gaps in hand-drawn lines so outer contour fills cleanly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(strokes)
    if contours:
        # Filter tiny noise specks
        valid = [c for c in contours if cv2.contourArea(c) > 64]
        if not valid:
            valid = contours
        cv2.drawContours(mask, valid, -1, 255, thickness=cv2.FILLED)
    else:
        mask = closed.copy()

    # Build "What the AI reads" visual overlay (dark slate bg, teal fill, neon green strokes)
    overlay = np.full((512, 512, 3), (24, 20, 18), dtype=np.uint8)  # BGR
    overlay[mask > 0] = (85, 65, 35)       # Subtle interior silhouette fill
    cnt_lines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnt_lines, -1, (220, 180, 70), 2)
    overlay[strokes > 0] = (130, 235, 95)  # Crisp bright green detected strokes

    return canvas, mask, strokes, overlay


def make_shaded_rgba(gray: np.ndarray, mask: np.ndarray, strokes: np.ndarray) -> Image.Image:
    """
    Creates a pseudo-3D shaded RGB+Alpha image from a 2D sketch so TripoSR/SF3D
    works well even in 'sketch' or 'plain' mode without Stable Diffusion.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = float(np.max(dist)) if np.max(dist) > 0 else 1.0
    norm_d = dist / max_d

    # Smooth dome shading + original sketch creases
    base = np.clip(155.0 + 75.0 * np.sqrt(norm_d), 0, 245)
    crease = (strokes.astype(np.float32) / 255.0) * 65.0
    shaded = np.clip(base - crease, 35, 245).astype(np.uint8)

    rgb = np.stack([shaded, shaded, shaded], axis=-1)
    rgba = np.dstack([rgb, mask])
    return Image.fromarray(rgba, mode="RGBA")


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Computes Intersection-over-Union (IoU) between two binary masks."""
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    a = mask_a > 127
    b = mask_b > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0