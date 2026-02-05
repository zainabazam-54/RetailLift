import cv2
from collections import defaultdict

class ShopliftingPipeline:

    def __init__(self, detector, classifier):
        self.detector = detector
        self.classifier = classifier
        # maintain history of predictions for each tracked person
        self.history = defaultdict(list)
        # maintain crop history for temporal sequences
        self.crop_history = defaultdict(list)
        self.SEQUENCE_LENGTH = 15

    def process_frame(self, frame):
        WINDOW_SIZE = 15

        detections = self.detector.detect(frame)

        for box, track_id in zip(detections.xyxy, detections.tracker_id):

            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]

            # 1️⃣ Build temporal sequence of crops
            self.crop_history[track_id].append(person_crop)
            if len(self.crop_history[track_id]) > self.SEQUENCE_LENGTH:
                self.crop_history[track_id].pop(0)

            # 2️⃣ Only predict when we have enough frames for temporal analysis
            if len(self.crop_history[track_id]) >= 10:
                _, prob = self.classifier.predict(self.crop_history[track_id])
            else:
                prob = 0.0


            # 3️⃣ Store prediction history
            self.history[track_id].append(prob)

            # 4️⃣ Keep only last 15 frames
            if len(self.history[track_id]) > WINDOW_SIZE:
                self.history[track_id].pop(0)

            # 5️⃣ Temporal smoothing
            avg_prob = sum(self.history[track_id]) / len(self.history[track_id]) if self.history[track_id] else 0

            # 6️⃣ Final decision - lower threshold since we now have temporal data
            if avg_prob > 0.5:
                label = "Shoplifting"
                color = (0, 0, 255)
            else:
                label = "Normal"
                color = (0, 255, 0)

            # 7️⃣ Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id}: {label} ({avg_prob:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                
                color,
                2
            )

        return frame
