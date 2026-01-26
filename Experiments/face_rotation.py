import numpy as np
import cv2

def rotate_face(image_path, landmarks):
    image = cv2.imread(image_path)

    left_eye = landmarks.get("landmarks").get("left_eye")
    right_eye = landmarks.get("landmarks").get("right_eye")

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx)) - 180

    eyes_center = [
        int((left_eye[0] + right_eye[0])/2),
        int((left_eye[1] + right_eye[1])/2)
    ]

    print("Rotation angle:", angle)
    cv2.circle(image, eyes_center, 5, (0,255,0), -1)

    M = cv2.getRotationMatrix2D(
        center=eyes_center,
        angle=angle,
        scale=1.0
    )

    aligned_image = cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    facial_area = landmarks.get("facial_area")
    new_facial_area = rotate_bbox(facial_area, M)

    cropped_face = aligned_image[new_facial_area[1]:new_facial_area[3], new_facial_area[0]:new_facial_area[2]]

    cv2.imwrite("Aligned_image.png", cropped_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate_bbox(facial_area, M):
    p1 = (facial_area[0], facial_area[1])
    p2 = (facial_area[2], facial_area[1])
    p3 = (facial_area[2], facial_area[3])
    p4 = (facial_area[0], facial_area[3])

    new_p1 = rotate_point(M, p1)
    new_p2 = rotate_point(M, p2)
    new_p3 = rotate_point(M, p3)
    new_p4 = rotate_point(M, p4)

    min_x = min([new_p1[0], new_p2[0], new_p3[0], new_p4[0]])
    max_x = max([new_p1[0], new_p2[0], new_p3[0], new_p4[0]])

    min_y = min([new_p1[1], new_p2[1], new_p3[1], new_p4[1]])
    max_y = max([new_p1[1], new_p2[1], new_p3[1], new_p4[1]])

    return min_x, min_y, max_x, max_y

def rotate_point(M, point):
    x, y = point
    new_x = M[0,0]*x + M[0,1]*y + M[0,2]
    new_y = M[1,0]*x + M[1,1]*y + M[1,2]
    return int(new_x), int(new_y)

img_path = r"P:\FaceTech\Face-Tech\Custom_Data\images\train\09265.png"
landmarks = {
        "score": 0.9705307483673096,
        "facial_area": [
            167,
            88,
            699,
            763
        ],
        "landmarks": {
            "right_eye": [
                270.06,
                387.78
            ],
            "left_eye": [
                515.03,
                333.29
            ],
            "nose": [
                406.59,
                513.54
            ],
            "mouth_right": [
                348.43,
                613.45
            ],
            "mouth_left": [
                565.41,
                568.61
            ]
        }
    }

rotate_face(img_path, landmarks)