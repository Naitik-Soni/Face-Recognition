import cv2
from test_face_recognition import is_same_face

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

FRAME_SKIP = 2  # Reduced for smoother real-time
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    
    if frame_id % FRAME_SKIP == 0:
        verified, distance, bbox, confidence = is_same_face(frame)
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if verified else (0, 0, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"Match={verified} dist={distance:.3f} conf={confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow("InsightFace Live Recognition (RTX 3050)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()