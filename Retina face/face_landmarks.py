# Import necessary libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from retinaface import RetinaFace

def get_face_landmarks(image_path):
    if not os.path.exists(image_path):
        raise ValueError("Image not found!")
    
    return RetinaFace.detect_faces(image_path)