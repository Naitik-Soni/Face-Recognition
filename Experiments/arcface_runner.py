import onnxruntime as ort
import numpy as np
from face_preprocessing import normalize_face_input
import cv2

sess = ort.InferenceSession(r"..\models\arc.onnx",
                            providers=["CPUExecutionProvider"])

def get_embedding(img_path):
    img = cv2.imread(img_path)
    x = normalize_face_input(img)

    input_name = sess.get_inputs()[0].name
    emb = sess.run(None, {input_name: x})[0]

    # L2 normalize (CRITICAL)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

emb1 = get_embedding("Aligned_image.png")
emb2 = get_embedding("Aligned_image2.png")

print(type(emb1), emb1.shape)

cosine_sim = np.dot(emb1, emb2.T)[0][0]
print("Cosine similarity:", cosine_sim)