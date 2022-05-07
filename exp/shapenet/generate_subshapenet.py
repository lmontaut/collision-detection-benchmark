import os
import hppfcl
import json
import pandas as pd
from pycolbench.utils import SHAPENET_PATH
from tqdm import trange

# Load ShapeNet meshes in subshapenet.json and get their number of vertices
shapenet_meshes_path = os.path.join(SHAPENET_PATH, "ShapeNetCore.v2")
with open(os.path.join(SHAPENET_PATH, 'subshapenet.json')) as f:
    meshes_json = json.load(f)
meshes_json = meshes_json["subshapenet_meshes"]

results = {}
fields = ["path", "num_points"]
for i in range(len(fields)):
    results[fields[i]] = []

loader = hppfcl.MeshLoader()
for i in trange(len(meshes_json)):
    category_path = os.path.join(shapenet_meshes_path, meshes_json[i]["id"])
    category_path_relative = os.path.join("ShapeNetCore.v2", meshes_json[i]["id"])
    subdirs = os.listdir(category_path)
    for j in range(len(subdirs)):
        absolute_path = os.path.join(os.path.join(category_path, subdirs[j]), "models/model_normalized.obj")
        relative_path = os.path.join(os.path.join(category_path_relative, subdirs[j]), "models/model_normalized.obj")
        mesh = loader.load(absolute_path)
        try:
            _ = mesh.buildConvexHull(True, "Qt")
            shape = mesh.convex
            results[fields[0]].append(relative_path)
            results[fields[1]].append(shape.num_points)
        except Exception as e:
            print(e)

print(f"Number of shapes in selected subshapenet: {len(results[fields[0]])}")
output_path = os.path.join(SHAPENET_PATH, "subshapenet.csv")
print("Saving results...")
df = pd.DataFrame(results, columns=fields)
df.to_csv(output_path)
print("Results saved.")
