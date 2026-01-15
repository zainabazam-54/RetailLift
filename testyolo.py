from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

video_path = "inout1.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Video Inference", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
