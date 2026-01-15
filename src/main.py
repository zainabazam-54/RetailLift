import cv2
from src.detector import PersonDetector
from src.classifier import ShopliftingClassifier
from src.pipeline import ShopliftingPipeline


detector=PersonDetector("src/models/best (2).pt")
classifier=ShopliftingClassifier("src/models/efficientNetModel.h5")
pipeline=ShopliftingPipeline(detector,classifier)

cap = cv2.VideoCapture("src/Normal (6).mp4")


print("Video opened:", cap.isOpened())

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break

    frame=pipeline.process_frame(frame)
    cv2.imshow("Shoplifting Detection",frame)

    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()