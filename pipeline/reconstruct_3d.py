import os
import sys
import numpy as np
import torch
import trimesh
from PIL import Image
from pipeline.inflate import finalize_mesh

_TSR_MODEL = None
_SF3D_MODEL = None

def get_device() -> str:
    override = os.environ.get("S2S_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_triposr():
    global _TSR_MODEL
    if _TSR_MODEL is not None:
        return _TSR_MODEL

    # Ensure third_party/TripoSR is on sys.path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tsr_dir = os.path.join(repo_root, "third_party", "TripoSR")
    if os.path.isdir(tsr_dir) and tsr_dir not in sys.path:
        sys.path.insert(0, tsr_dir)

    from tsr.system import TSR
    device = get_device()
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    _TSR_MODEL = model
    return _TSR_MODEL


def prepare_rgba_for_triposr(rgba_pil: Image.Image) -> Image.Image:
    """Composites RGBA foreground onto neutral 0.5 gray background as required by TripoSR."""
    arr = np.array(rgba_pil).astype(np.float32) / 255.0
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3:4]
        comp = rgb * alpha + 0.5 * (1.0 - alpha)
        arr_uint8 = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="RGB").resize((512, 512))
    return rgba_pil.convert("RGB").resize((512, 512))


def orient_for_gltf(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotates TripoSR output so +Y is up and +Z is front in glTF / model-viewer."""
    rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    mesh.apply_transform(rot_y @ rot_x)
    return finalize_mesh(mesh)


def reconstruct_mesh(rgba_pil: Image.Image, backend: str = "triposr", resolution: int = 192):
    """
    Runs 3D reconstruction using TripoSR (or SF3D if available).
    Returns (trimesh.Trimesh, actual_backend_used_str).
    """
    device = get_device()

    # Optional SF3D path if installed in third_party/stable-fast-3d
    if backend == "sf3d" and device == "cuda":
        try:
            global _SF3D_MODEL
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            sf3d_dir = os.path.join(repo_root, "third_party", "stable-fast-3d")
            if os.path.isdir(sf3d_dir) and sf3d_dir not in sys.path:
                sys.path.insert(0, sf3d_dir)
            from sf3d.system import SF3D
            if _SF3D_MODEL is None:
                _SF3D_MODEL = SF3D.from_pretrained(
                    "stabilityai/stable-fast-3d",
                    config_name="config.yaml",
                    weight_name="model.safetensors"
                ).to(device)
                _SF3D_MODEL.eval()
            with torch.no_grad():
                mesh, _ = _SF3D_MODEL.run_image(rgba_pil, bake_resolution=512, remesh_option="none")
            return finalize_mesh(mesh), "SF3D (GPU)"
        except Exception as e:
            print(f"[INFO] SF3D unavailable ({e}), falling back to TripoSR.")

    # Main TripoSR reconstruction
    model = load_triposr()
    inp_rgb = prepare_rgba_for_triposr(rgba_pil)
    with torch.no_grad():
        scene_codes = model([inp_rgb], device=device)
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=resolution)
    mesh = orient_for_gltf(meshes[0])
    return mesh, f"TripoSR ({device.upper()})"