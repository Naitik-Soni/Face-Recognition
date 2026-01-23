from face_landmarks import get_face_landmarks
from face_landmarks_drawer import draw_face_landmarks
from utils import save_landmarks_to_json
import os

if not os.path.exists(r".\Outputs"):
    os.makedirs(r".\Outputs")

def main(IMAGE_PATH, RESIZE_FACTOR=0.6, SAVE_FACE_LANDMARKS=False):
    IMAGE_PATH = r"C:\Users\baps\Documents\Projects\Face-Recognition\Custom_Data\images\train\09992.png"

    base_name = os.path.basename(IMAGE_PATH)
    IMAGE_SAVE_PATH = fr".\Outputs\{base_name}"
    RESIZE_FACTOR = 0.6

    # Get face landmarks
    face_landmarks = get_face_landmarks(IMAGE_PATH)
    total_faces = len(face_landmarks.keys())
    print(F"Extracted {total_faces} face landmarks")

    # Draw and save face landmarks
    draw_face_landmarks(IMAGE_PATH, face_landmarks, IMAGE_SAVE_PATH, RESIZE_FACTOR)
    save_landmarks_to_json(face_landmarks, IMAGE_SAVE_PATH.replace('.png', '_landmarks.json'))

    if SAVE_FACE_LANDMARKS:
        print(f"Landmarks drawn and saved to ./Outputs for {base_name}")
    else:
        print(f"Drawn landmarks for image {base_name}")


if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\baps\Documents\Projects\Face-Recognition\Custom_Data\images\train\09992.png"
    main(IMAGE_PATH, RESIZE_FACTOR=0.6, SAVE_FACE_LANDMARKS=True)