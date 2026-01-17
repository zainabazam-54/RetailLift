# src/pipeline.py
import cv2
from collections import defaultdict

class ShopliftingPipeline:

    def __init__(self, detector, classifier):
        self.detector = detector
        self.classifier = classifier
        # maintain history of predictions for each tracked person
        self.history = defaultdict(list)

    def process_frame(self, frame):
        WINDOW_SIZE = 15

        detections = self.detector.detect(frame)

        for box, track_id in zip(detections.xyxy, detections.tracker_id):

            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]

            # 1️⃣ Get classifier prediction (probability 0–1)
            _, prob = self.classifier.predict(person_crop)


            # 2️⃣ Store prediction history
            self.history[track_id].append(prob)

            # 3️⃣ Keep only last 15 frames
            if len(self.history[track_id]) > WINDOW_SIZE:
                self.history[track_id].pop(0)

            # 4️⃣ Temporal smoothing
            avg_prob = sum(self.history[track_id]) / len(self.history[track_id])

            # 5️⃣ Final decision
            if avg_prob > 0.7:
                label = "Shoplifting"
                color = (0, 0, 255)
            else:
                label = "Normal"
                color = (0, 255, 0)

            # 6️⃣ Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id}: {label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                
                color,
                2
            )

        return frame
