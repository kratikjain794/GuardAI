import os
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel

from app.ai.emotion_detection import EmotionDetector
from app.ai.voice_detection import VoiceDistressDetector
from app.ai.camera_detection import CameraDetector


# ==========================================
# Router
# ==========================================

router = APIRouter(
    prefix="/monitoring",
    tags=["Safety Monitoring"],
)


# ==========================================
# Monitoring Settings
# ==========================================

class MonitoringSettings(BaseModel):
    voice_detection: bool = True
    emotion_detection: bool = True
    camera_detection: bool = False


# ==========================================
# Monitoring State
# ==========================================

monitoring_state = {
    "active": False,
    "voice_detection": False,
    "emotion_detection": False,
    "camera_detection": False,
}


# ==========================================
# Get Monitoring Status
# ==========================================

@router.get("/status")
async def get_monitoring_status():

    return monitoring_state


# ==========================================
# Start Monitoring
# ==========================================

@router.post("/start")
async def start_monitoring(
    settings: MonitoringSettings,
):

    monitoring_state["active"] = True

    monitoring_state["voice_detection"] = (
        settings.voice_detection
    )

    monitoring_state["emotion_detection"] = (
        settings.emotion_detection
    )

    monitoring_state["camera_detection"] = (
        settings.camera_detection
    )

    return {
        "status": "success",
        "message": "Safety monitoring started",
        "monitoring": monitoring_state,
    }


# ==========================================
# Stop Monitoring
# ==========================================

@router.post("/stop")
async def stop_monitoring():

    monitoring_state["active"] = False
    monitoring_state["voice_detection"] = False
    monitoring_state["emotion_detection"] = False
    monitoring_state["camera_detection"] = False

    return {
        "status": "success",
        "message": "Safety monitoring stopped",
        "monitoring": monitoring_state,
    }


# ==========================================
# Emotion Detection
# ==========================================

@router.post("/emotion")
async def detect_emotion(
    audio: UploadFile = File(...),
):

    filename = audio.filename or ""

    # --------------------------------------
    # Validate file
    # --------------------------------------

    if Path(filename).suffix.lower() != ".wav":

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only WAV audio files are supported.",
        )

    temp_path = None

    try:

        # ----------------------------------
        # Read uploaded audio
        # ----------------------------------

        audio_data = await audio.read()

        if not audio_data:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty.",
            )

        # ----------------------------------
        # Save temporary audio
        # ----------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as temp_file:

            temp_file.write(audio_data)

            temp_path = temp_file.name

        # ----------------------------------
        # Emotion Prediction
        # ----------------------------------

        detector = EmotionDetector()

        result = detector.predict(
            temp_path
        )

        # ----------------------------------
        # Response
        # ----------------------------------

        return {
            "status": "success",
            "filename": filename,
            "emotion": result["emotion"],
            "confidence": result["confidence"],
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Emotion detection failed: "
                f"{error}"
            ),
        )

    finally:

        await audio.close()

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)


# ==========================================
# Voice Distress Detection
# ==========================================

@router.post("/distress")
async def detect_distress(
    audio: UploadFile = File(...),
):

    filename = audio.filename or ""

    # --------------------------------------
    # Validate file
    # --------------------------------------

    if Path(filename).suffix.lower() != ".wav":

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only WAV audio files are supported.",
        )

    temp_path = None

    try:

        # ----------------------------------
        # Read uploaded audio
        # ----------------------------------

        audio_data = await audio.read()

        if not audio_data:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty.",
            )

        # ----------------------------------
        # Save temporary audio
        # ----------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as temp_file:

            temp_file.write(audio_data)

            temp_path = temp_file.name

        # ----------------------------------
        # Distress Prediction
        # ----------------------------------

        detector = VoiceDistressDetector()

        result = detector.predict(
            temp_path
        )

        # ----------------------------------
        # Response
        # ----------------------------------

        return {
            "status": "success",
            "filename": filename,
            "distress_detected": result[
                "distress_detected"
            ],
            "label": result["label"],
            "confidence": result["confidence"],
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Distress detection failed: "
                f"{error}"
            ),
        )

    finally:

        await audio.close()

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)


# ==========================================
# Camera Person Detection
# ==========================================

@router.post("/camera")
async def detect_camera(
    image: UploadFile = File(...),
):

    filename = image.filename or ""

    extension = Path(
        filename
    ).suffix.lower()

    # --------------------------------------
    # Validate image type
    # --------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are supported."
            ),
        )

    temp_path = None

    try:

        # ----------------------------------
        # Read uploaded image
        # ----------------------------------

        image_data = await image.read()

        if not image_data:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty.",
            )

        # ----------------------------------
        # Save temporary image
        # ----------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temp_file.write(
                image_data
            )

            temp_path = (
                temp_file.name
            )

        # ----------------------------------
        # Person Detection
        # ----------------------------------

        detector = CameraDetector()

        result = detector.detect(
            temp_path
        )

        # ----------------------------------
        # Response
        # ----------------------------------

        return {
            "status": "success",
            "filename": filename,

            "person_detected": result[
                "person_detected"
            ],

            "person_count": result[
                "person_count"
            ],

            "people": result[
                "people"
            ],
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Camera detection failed: "
                f"{error}"
            ),
        )

    finally:

        await image.close()

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)