# Import necessary libraries
import cv2
import random

def draw_face_landmarks(image_path, data, save_path=None):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found!")
    
    h, w = img.shape[:2]
    r_factor = w/1280

    # Draw all faces
    for face_id, face_data in data.items():
        # Random color per face
        color = tuple(random.randint(150, 255) for _ in range(3))

        # Bounding box
        x1, y1, x2, y2 = map(int, face_data["facial_area"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, int(5*r_factor))

        # Face label
        cv2.putText(
            img,
            face_id,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            r_factor,
            color,
            3
        )

        # Landmarks
        for name, (x, y) in face_data["landmarks"].items():
            x, y = int(x), int(y)

            # Point
            cv2.circle(img, (x, y), int(12*r_factor), (0, 0, 255), -1)

            # Label
            cv2.putText(
                img,
                name,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                r_factor,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    # Show result
    cv2.imwrite(save_path, img)