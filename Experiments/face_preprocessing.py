import cv2
import numpy as np

# face_path = r"P:\FaceTech\Face-Tech\Retina face\Aligned_image.png"

def normalize_face_input(face):
    # Resize
    face = cv2.resize(face, (112, 112))

    img = face.astype(np.float32)

    # BGR → RGB
    img = img[:, :, ::-1]

    # Normalize to [-1, 1]
    img = (img - 127.5) / 128.0

    # Add batch dimension ONLY (NHWC)
    img = np.expand_dims(img, axis=0)

    # Ensure contiguous
    img = np.ascontiguousarray(img)

    return img
