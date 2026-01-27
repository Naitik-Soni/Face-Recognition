from deepface import DeepFace
import numpy as np

# Returns a list of embeddings, faces, and locations
result = DeepFace.represent(img_path=r"P:\FaceTech\Face-Tech\Experiments\Aligned_image.png", model_name="ArcFace")

embedding = result[0]["embedding"]

em1 = np.array(embedding, dtype=np.float32)

em1 = em1 / np.linalg.norm(em1)

cosine_sim = np.dot(em1, em1)
print(cosine_sim)