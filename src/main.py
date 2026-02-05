import cv2
from detector import PersonDetector

from classifier import ShopliftingClassifier
from pipeline import ShopliftingPipeline

import torch
import torch.nn as nn

class CNNGRU(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1):
        super(CNNGRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.gru = nn.GRU(
            input_size=64 * 4 * 4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        cnn_out = []
        for t in range(seq_len):
            frame_feat = self.cnn(x[:, t])
            frame_feat = frame_feat.reshape(batch_size, -1)
            cnn_out.append(frame_feat)
        cnn_out = torch.stack(cnn_out, dim=1)
        gru_out, _ = self.gru(cnn_out)
        out = self.fc(gru_out[:, -1, :])
        return out.squeeze(1)

detector=PersonDetector("src/models/best (2).pt")
classifier=ShopliftingClassifier("src/models/efficientNetModel.h5")
pipeline=ShopliftingPipeline(detector,classifier)

cap = cv2.VideoCapture("Shoplifting007_x264_4.mp4")


print("Video opened:", cap.isOpened())

frame_number = 0

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break

    frame=pipeline.process_frame(frame)
    cv2.imshow("Shoplifting Detection",frame)

    frame_number += 1

    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
