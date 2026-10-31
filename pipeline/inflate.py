import cv2
import numpy as np
import trimesh
from skimage.measure import marching_cubes

def mesh_by_inflation(mask: np.ndarray, thickness: int = 20) -> trimesh.Trimesh:
    """Inflates a 2D mask into a 3D mesh using distance transforms."""
    # Calculate depth based on distance from the edge
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist = (dist / (np.max(dist) + 1e-5)) * thickness
    
    # Create a 3D volume bounding box (Z, Y, X)
    h, w = mask.shape
    depth = int(thickness * 2) + 4
    volume = np.zeros((depth, h, w), dtype=np.float32)
    
    # Project the 2.5D distance transform onto the front and back of the volume
    mid = depth // 2
    for z in range(depth):
        # Falloff distance from the center z-plane
        z_dist = abs(z - mid)
        # Only keep voxels where the 2D distance is greater than the Z distance
        volume[z] = np.maximum(0, dist - z_dist)
        
    # Extract the mesh using marching cubes
    verts, faces, normals, values = marching_cubes(volume, level=0.1)
    
    # Build and orient the mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Center and scale to a 1x1x1 unit box for export
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / np.max(mesh.extents))
    
    # Rotate to stand upright (depending on coordinate system)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    
    return mesh