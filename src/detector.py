from ultralytics import YOLO
import supervision as sv

class PersonDetector:
    def __init__(self, yolo_path):
        self.model = YOLO("src/models/best (2).pt")
        self.tracker = sv.ByteTrack()

    def detect(self, frame):
        result = self.model(frame, conf=0.4, classes=[0])[0]  #bounding boxes generation
        detections = sv.Detections.from_ultralytics(result)     #convert to supervision format
        tracked = self.tracker.update_with_detections(detections) #track the detected persons
        return tracked