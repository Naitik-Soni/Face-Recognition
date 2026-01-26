import cv2
import numpy as np

# face_path = r"P:\FaceTech\Face-Tech\Retina face\Aligned_image.png"

def normalize_face_input(face_path):
    face = cv2.imread(face_path)

    # Resize face
    face = cv2.resize(face, (112, 112))

    # Change the color channel
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Convert to type float 32
    face = face.astype(np.float32)

    # Normalize the range (-1 to +1)
    face = (face - 127.5) / 127.5

    # Transform for ArcFace input (channel, height, width)
    face = np.transpose(face, (2, 0, 1))

    # Expand the dimensions (batch, channels, height, width)
    face = np.expand_dims(face, axis=0)

    return face