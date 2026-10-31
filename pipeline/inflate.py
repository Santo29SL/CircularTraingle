import cv2
import numpy as np
import trimesh
from skimage.measure import marching_cubes

def mesh_by_inflation(mask: np.ndarray, resolution: int = 160, thickness: float = 28.0) -> trimesh.Trimesh:
    """Inflates a 2D binary silhouette into a smooth symmetric 3D mesh."""
    small = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    dist = cv2.distanceTransform(small, cv2.DIST_L2, 5)
    max_d = float(np.max(dist)) if np.max(dist) > 0 else 1.0

    # Elliptical profile (sqrt falloff gives rounded 3D surfaces instead of sharp cones)
    profile = np.sqrt(np.clip(dist / max_d, 0.0, 1.0)) * thickness

    h, w = small.shape
    depth = int(thickness * 2) + 8
    volume = np.zeros((w, h, depth), dtype=np.float32)  # (X, Y, Z)

    mid = depth // 2
    # Flip Y so image top is +Y in 3D
    prof_xy = np.flipud(profile).T  # shape (X, Y)
    for z in range(depth):
        z_dist = abs(z - mid)
        volume[:, :, z] = np.maximum(0.0, prof_xy - z_dist)

    verts, faces, normals, _ = marching_cubes(volume, level=0.25)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    return finalize_mesh(mesh)


def finalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centers, scales to unit cube, and ensures valid geometry."""
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) > 0:
        mesh.apply_translation(-mesh.centroid)
        max_ext = float(np.max(mesh.extents))
        if max_ext > 1e-6:
            mesh.apply_scale(1.0 / max_ext)
    return mesh