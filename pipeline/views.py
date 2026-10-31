import cv2
import numpy as np
import trimesh
from skimage.measure import marching_cubes
from pipeline.preprocess import compute_iou

def render_orthographic_silhouette(mesh: trimesh.Trimesh, axis_x: int, axis_y: int, flip_x=False, flip_y=True, size=256) -> np.ndarray:
    """Renders a binary silhouette of a normalized mesh along a chosen pair of axes using OpenCV."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    if mesh.is_empty or len(mesh.faces) == 0:
        return canvas

    verts = mesh.vertices.copy()
    # Normalize to [-0.42, 0.42] based on overall max extent
    center = verts.mean(axis=0)
    verts -= center
    scale = np.max(np.ptp(verts, axis=0))
    if scale > 1e-6:
        verts = verts / scale * 0.82

    pts_x = -verts[:, axis_x] if flip_x else verts[:, axis_x]
    pts_y = -verts[:, axis_y] if flip_y else verts[:, axis_y]

    px = ((pts_x + 0.5) * size).astype(np.int32)
    py = ((pts_y + 0.5) * size).astype(np.int32)
    pts_2d = np.stack([px, py], axis=-1)

    tris = pts_2d[mesh.faces]
    for tri in tris:
        cv2.fillConvexPoly(canvas, tri, 255)

    return canvas


def render_six_views_and_iou(mesh: trimesh.Trimesh, input_mask: np.ndarray, out_path: str) -> float:
    """
    Renders a 6-panel orthographic montage (Front, Back, Left, Right, Top, Bottom)
    and returns Front Silhouette IoU vs the user's input sketch mask.
    """
    # Standard glTF coordinates: X=Right, Y=Up, Z=Front
    views = [
        ("FRONT (Input Match)", render_orthographic_silhouette(mesh, 0, 1, flip_x=False, flip_y=True)),
        ("BACK (Predicted)",    render_orthographic_silhouette(mesh, 0, 1, flip_x=True,  flip_y=True)),
        ("LEFT (Predicted)",    render_orthographic_silhouette(mesh, 2, 1, flip_x=False, flip_y=True)),
        ("RIGHT (Predicted)",   render_orthographic_silhouette(mesh, 2, 1, flip_x=True,  flip_y=True)),
        ("TOP (Predicted)",     render_orthographic_silhouette(mesh, 0, 2, flip_x=False, flip_y=False)),
        ("BOTTOM (Predicted)",  render_orthographic_silhouette(mesh, 0, 2, flip_x=False, flip_y=True)),
    ]

    front_sil = views[0][1]
    front_iou = compute_iou(input_mask, front_sil)

    # Create 2x3 dark studio montage image (3 columns x 2 rows)
    cell_w, cell_h = 220, 240
    grid = np.full((cell_h * 2, cell_w * 3, 3), (22, 19, 17), dtype=np.uint8)

    for idx, (label, sil) in enumerate(views):
        r, c = divmod(idx, 3)
        y0, x0 = r * cell_h, c * cell_w

        # Resize silhouette to 180x180
        sil_small = cv2.resize(sil, (180, 180), interpolation=cv2.INTER_NEAREST)
        panel = np.full((180, 180, 3), (34, 30, 27), dtype=np.uint8)
        panel[sil_small > 0] = (235, 235, 235)

        # Draw border
        cv2.rectangle(panel, (0, 0), (179, 179), (65, 60, 55), 1)
        grid[y0 + 38:y0 + 218, x0 + 20:x0 + 200] = panel

        # Title text
        title = f"FRONT (IoU: {front_iou:.2f})" if idx == 0 else label
        color = (130, 235, 95) if idx == 0 else (180, 180, 180)
        cv2.putText(grid, title, (x0 + 20, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, grid)
    return front_iou


def carve_mesh_with_side_silhouette(mesh: trimesh.Trimesh, side_mask: np.ndarray, view_side: str = "right", res: int = 96) -> trimesh.Trimesh:
    """
    Visual-hull voxel carving: voxelizes the existing mesh and removes voxels
    outside the user's drawn side-view silhouette.
    """
    verts = mesh.vertices.copy()
    center = verts.mean(axis=0)
    verts -= center
    max_ext = float(np.max(np.ptp(verts, axis=0)))
    if max_ext < 1e-6:
        return mesh
    verts = verts / max_ext  # in [-0.5, 0.5]

    norm_mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy())
    pitch = 1.0 / res
    vox = norm_mesh.voxelized(pitch=pitch).fill()
    mat = vox.matrix.astype(np.float32)  # shape (nx, ny, nz)

    # Resize side_mask to match (nz, ny)
    nx, ny, nz = mat.shape
    mask_resized = cv2.resize(side_mask, (nz, ny), interpolation=cv2.INTER_NEAREST)  # rows=Y (flipped), cols=Z
    mask_zy = (np.flipud(mask_resized) > 127).T  # shape (nz, ny) -> transpose to (ny, nz)
    if view_side == "left":
        mask_zy = np.fliplr(mask_zy)

    # Broadcast across X axis: keep only voxels where side silhouette is active
    keep = mask_zy[None, :, :]  # (1, ny, nz)
    carved_vol = mat * keep

    if np.sum(carved_vol > 0.5) < 32:
        # If carving removed almost everything, return original mesh
        return mesh

    padded = np.pad(carved_vol, 2, mode="constant")
    v, f, n, _ = marching_cubes(padded, level=0.5)
    out = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n)
    out.apply_translation(-out.centroid)
    out.apply_scale(1.0 / max(np.max(out.extents), 1e-5))
    return out