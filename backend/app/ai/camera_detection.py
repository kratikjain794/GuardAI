from pathlib import Path

from ultralytics import YOLO


# ==========================================
# Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "yolo11n.pt"


# ==========================================
# Configuration
# ==========================================

CONFIDENCE_THRESHOLD = 0.40


# ==========================================
# Camera / Human Detector
# ==========================================

class CameraDetector:

    def __init__(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"YOLO model not found:\n{MODEL_PATH}"
            )

        print(
            f"Loading Human Detection Model:\n"
            f"{MODEL_PATH}"
        )

        self.model = YOLO(
            str(MODEL_PATH)
        )

        print(
            "Human Detection Model Loaded Successfully!"
        )


    # ======================================
    # Detect Humans
    # ======================================

    def detect(
        self,
        image_path,
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        results = self.model.predict(

            source=str(image_path),

            conf=CONFIDENCE_THRESHOLD,

            verbose=False,

        )

        people = []


        # ==================================
        # Process Results
        # ==================================

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


                # --------------------------
                # Person Only
                # --------------------------

                if class_name != "person":
                    continue


                confidence = float(
                    box.conf.item()
                )


                coordinates = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .tolist()
                )


                people.append({

                    "confidence":
                        round(
                            confidence * 100,
                            2,
                        ),

                    "bounding_box": {

                        "x1":
                            round(
                                coordinates[0],
                                2,
                            ),

                        "y1":
                            round(
                                coordinates[1],
                                2,
                            ),

                        "x2":
                            round(
                                coordinates[2],
                                2,
                            ),

                        "y2":
                            round(
                                coordinates[3],
                                2,
                            ),
                    },
                })


        # ==================================
        # Final Result
        # ==================================

        person_count = len(
            people
        )


        return {

            "person_detected":
                person_count > 0,

            "person_count":
                person_count,

            "people":
                people,

        }


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "GuardAI Human Detection"
    )

    print(
        "=============================="
    )


    detector = CameraDetector()


    image_path = input(
        "\nEnter image path: "
    ).strip()


    result = detector.detect(
        image_path
    )


    print(
        "\n=============================="
    )

    print(
        "Detection Result"
    )

    print(
        "=============================="
    )


    print(
        f"Person Detected : "
        f"{result['person_detected']}"
    )

    print(
        f"Person Count    : "
        f"{result['person_count']}"
    )


    for index, person in enumerate(
        result["people"],
        start=1,
    ):

        print(
            f"\nPerson {index}"
        )

        print(
            f"Confidence : "
            f"{person['confidence']}%"
        )

        print(
            f"Bounding Box : "
            f"{person['bounding_box']}"
        )