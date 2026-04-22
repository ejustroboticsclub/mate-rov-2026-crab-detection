import os

import cv2

from .utils import get_model


class CrabDetector:
    def __init__(self,output_dir="./output"):
        self.model = get_model()
        self.TARGET_CLASSES = ["green-crab", "rock-crab", "jonah-crab"]
        self.CONF_THRESHOLD = 0.5
        self.CLASS_COLORS = {
            "green-crab": (0, 255, 0),
            "rock-crab": (255, 0, 0),
            "jonah-crab": (0, 0, 255),
        }
        self.totalCount = 0
        self.OUTPUT_DIR = output_dir 

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def detect(self, imgPath) -> tuple:
        frame = cv2.imread(imgPath)
        results = self.model.predict(frame)
        stats = {CLS: 0 for CLS in self.TARGET_CLASSES}
        for result in results:
            for box, cls, conf in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            ):
                class_name = self.model.names[int(cls)]
                if conf < self.CONF_THRESHOLD or not class_name in self.TARGET_CLASSES:
                    continue

                self.totalCount += 1
                stats[class_name] += 1
                
                x1, y1, x2, y2 = map(int,box)
                label = f"{class_name} {conf:.2f}"
                color = self.CLASS_COLORS.get(class_name, (0, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        cv2.putText(
                    frame,
                    f"Green Crabs: {stats['green-crab']}",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )

        cv2.putText(
                    frame,
                    f"Total: {self.totalCount}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
        return frame, self.totalCount

