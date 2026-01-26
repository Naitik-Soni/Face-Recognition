from face_landmarks import get_face_landmarks
from face_landmarks_drawer import draw_face_landmarks
from utils import save_landmarks_to_json, get_image_paths
import os

if not os.path.exists(r".\Outputs"):
    os.makedirs(r".\Outputs")

def main(IMAGE_PATH, SAVE_FACE_LANDMARKS=False):
    base_name = os.path.basename(IMAGE_PATH)
    IMAGE_SAVE_PATH = fr".\Outputs\{base_name}"

    # Get face landmarks
    face_landmarks = get_face_landmarks(IMAGE_PATH)
    total_faces = len(face_landmarks.keys())
    print(F"Extracted {total_faces} face landmarks")

    # Draw and save face landmarks
    draw_face_landmarks(IMAGE_PATH, face_landmarks, IMAGE_SAVE_PATH)

    if SAVE_FACE_LANDMARKS:
        save_landmarks_to_json(face_landmarks, IMAGE_SAVE_PATH.replace('.png', '_landmarks.json'))
        print(f"Landmarks drawn and saved to ./Outputs for {base_name}")
    else:
        print(f"Drawn landmarks for image {base_name}")


if __name__ == "__main__":
    INPUT_DIR = r"../Custom_Data/images/train"

    images_path = get_image_paths(INPUT_DIR)

    images_path = [
        r"P:\FaceTech\Face-Tech\Custom_Data\images\train\09265.png"
    ]

    for image_path in images_path:
        main(image_path, SAVE_FACE_LANDMARKS=True)