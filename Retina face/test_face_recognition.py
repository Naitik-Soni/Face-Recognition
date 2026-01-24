import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# Force GPU + specific modules (your config is perfect)
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition", "genderage"],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # ADD THIS
)

app.prepare(ctx_id=0, det_size=(640, 640))  # GPU ctx_id=0

# Reference embedding (unchanged)
REFERENCE_IMG = r"C:\Users\naiti\Downloads\rishi.jpeg"
ref_img = cv2.imread(REFERENCE_IMG)
if ref_img is None:
    raise ValueError("Reference image not found")

ref_faces = app.get(ref_img)
if not ref_faces:
    raise ValueError("No face detected in reference image")

ref_embedding = ref_faces[0].embedding / norm(ref_faces[0].embedding)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))  # FIX: Proper L2-normalized cosine

def is_same_face(frame, threshold=0.35):  # 0.35 is good for single reference
    faces = app.get(frame)
    
    if not faces:
        return False, 1.0, None
    
    # Use highest confidence face
    face = max(faces, key=lambda f: f.det_score)
    emb = face.embedding / norm(face.embedding)
    
    dist = cosine_distance(ref_embedding, emb)
    verified = dist < threshold
    
    return verified, float(dist), face.bbox.astype(int), face.det_score