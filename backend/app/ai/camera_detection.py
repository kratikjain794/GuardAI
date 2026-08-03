from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "trained_models"

YOLO_MODEL = "yolo11n.pt"


class CameraDetector:

    def __init__(self):

        # YOLO model.
        # First run may download pretrained weights.
        self.model = YOLO(YOLO_MODEL)

    def detect(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=0.40,
            verbose=False,
        )

        people = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls.item()
                )

                class_name = (
                    self.model.names[class_id]
                )

                # Only detect people
                if class_name != "person":
                    continue

                confidence = float(
                    box.conf.item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                people.append(
                    {
                        "confidence": round(
                            confidence * 100,
                            2,
                        ),
                        "bounding_box": {
                            "x1": round(
                                coordinates[0],
                                2,
                            ),
                            "y1": round(
                                coordinates[1],
                                2,
                            ),
                            "x2": round(
                                coordinates[2],
                                2,
                            ),
                            "y2": round(
                                coordinates[3],
                                2,
                            ),
                        },
                    }
                )

        person_count = len(people)

        return {
            "person_detected":
                person_count > 0,

            "person_count":
                person_count,

            "people":
                people,
        }