from .utils import get_model
import os
import cv2

class CrabDetector:
    def __init__(self):
        self.model = get_model()
        self.TARGET_CLASS = "green_crab"
        self.OUTPUT_DIR = "tests/output"

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def detect(self, imagePath) -> tuple:
        results = self.model.predict(source=imagePath, save=False, verbose=False)
        
        img = cv2.imread(imagePath)
        if img is None:
            return None, 0
            
        h, w, _ = img.shape
        total_count = 0
        font_scale = w / 1000 
        thickness = max(1, int(font_scale * 2))

        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                class_name = self.model.names[int(cls)]

                if class_name == self.TARGET_CLASS:
                    total_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(img, label, (x1, max(0, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (0, 255, 0), thickness)

        count_text = f"Total {self.TARGET_CLASS}: {total_count}"
        cv2.putText(img, count_text, (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness + 1)

        base_name = os.path.splitext(os.path.basename(imagePath))[0]
        outputPath = os.path.join(self.OUTPUT_DIR, f"{base_name}_detection.png")
        cv2.imwrite(outputPath, img)

        return img, total_count
    
# detector = CrabDetector()
# image, count = detector.detect("SAMPLE_PATH")
