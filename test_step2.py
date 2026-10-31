import os
from pipeline.preprocess import extract_silhouette
from pipeline.inflate import mesh_by_inflation

print("1. Extracting silhouette...")
mask = extract_silhouette("test.png")

print("2. Inflating to 3D mesh...")
mesh = mesh_by_inflation(mask)

print("3. Exporting to GLB...")
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/test_inflate.glb"
mesh.export(output_path)

print(f"✅ Success! Open {output_path} in a 3D viewer.")