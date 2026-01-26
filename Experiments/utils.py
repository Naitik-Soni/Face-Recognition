import json
import numpy as np
import os
import glob

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_landmarks_to_json(data, save_path):
    """Saves the face landmarks dictionary to a JSON file."""

    # Optional: round landmark points
    for face in data.values():
        for k, v in face["landmarks"].items():
            face["landmarks"][k] = [round(float(x), 2) for x in v]

    # Save FULL data (not just last face)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=numpy_to_python)

def get_image_paths(dir_path, extensions=("*.jpg", "*.png", "*.jpeg")):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    return sorted(paths)
