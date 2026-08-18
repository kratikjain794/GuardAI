import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.ai.emotion_detection import EmotionDetector
from app.ai.voice_detection import VoiceDistressDetector
from app.ai.speech_keyword_detection import SpeechKeywordDetector
from app.ai.camera_detection import CameraDetector

from app.routes.risk import calculate_risk

from app.database.database import (
    camera_detections_collection,
)


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
# Camera Upload Directory
# ==========================================

BACKEND_DIR = Path(
    __file__
).resolve().parents[2]

CAMERA_UPLOAD_DIR = (
    BACKEND_DIR
    / "uploads"
    / "camera"
)

CAMERA_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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

    if (
        Path(filename)
        .suffix
        .lower()
        != ".wav"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Only WAV audio files "
                "are supported."
            ),
        )

    temp_path = None

    try:

        audio_data = await audio.read()

        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Uploaded audio file "
                    "is empty."
                ),
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as temp_file:

            temp_file.write(audio_data)
            temp_path = temp_file.name

        detector = EmotionDetector()

        result = detector.predict(
            temp_path
        )

        return {
            "status": "success",
            "filename": filename,
            "emotion": result.get(
                "emotion"
            ),
            "confidence": result.get(
                "confidence",
                0,
            ),
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                f"Emotion detection failed: "
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

    if (
        Path(filename)
        .suffix
        .lower()
        != ".wav"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Only WAV audio files "
                "are supported."
            ),
        )

    temp_path = None

    try:

        audio_data = await audio.read()

        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Uploaded audio file "
                    "is empty."
                ),
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as temp_file:

            temp_file.write(audio_data)
            temp_path = temp_file.name

        detector = (
            VoiceDistressDetector()
        )

        result = detector.predict(
            temp_path
        )

        return {
            "status": "success",
            "filename": filename,
            "distress_detected": bool(
                result.get(
                    "distress_detected",
                    False,
                )
            ),
            "label": result.get(
                "label"
            ),
            "confidence": result.get(
                "confidence",
                0,
            ),
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                f"Distress detection failed: "
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

    filename = (
        image.filename
        or "camera_capture.jpg"
    )

    extension = Path(
        filename
    ).suffix.lower()

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

    saved_path = None

    try:

        # Read image
        image_data = await image.read()

        if not image_data:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Uploaded image is empty."
                ),
            )

        # Unique filename
        unique_name = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex}"
            f"{extension}"
        )

        saved_path = (
            CAMERA_UPLOAD_DIR
            / unique_name
        )

        # Save permanently
        with open(
            saved_path,
            "wb",
        ) as saved_file:

            saved_file.write(
                image_data
            )

        # AI detection
        detector = CameraDetector()

        result = detector.detect(
            str(saved_path)
        )

        # MongoDB record
        detection_document = {
            "original_filename": filename,

            "saved_filename": unique_name,

            "image_path": (
                f"uploads/camera/"
                f"{unique_name}"
            ),

            "person_detected": bool(
                result.get(
                    "person_detected",
                    False,
                )
            ),

            "person_count": int(
                result.get(
                    "person_count",
                    0,
                )
            ),

            "people": result.get(
                "people",
                [],
            ),

            "created_at": datetime.now(
                timezone.utc
            ),
        }

        db_result = (
            await camera_detections_collection.insert_one(
                detection_document
            )
        )

        return {
            "status": "success",

            "detection_id": str(
                db_result.inserted_id
            ),

            "filename": filename,

            "saved_filename": unique_name,

            "image_url": (
                "/monitoring/camera/image/"
                f"{unique_name}"
            ),

            "person_detected": bool(
                result.get(
                    "person_detected",
                    False,
                )
            ),

            "person_count": int(
                result.get(
                    "person_count",
                    0,
                )
            ),

            "people": result.get(
                "people",
                [],
            ),
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        if (
            saved_path
            and saved_path.exists()
        ):
            saved_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        )

    except Exception as error:

        if (
            saved_path
            and saved_path.exists()
        ):
            saved_path.unlink()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                f"Camera detection failed: "
                f"{error}"
            ),
        )

    finally:

        await image.close()


# ==========================================
# Camera Detection History
# ==========================================

@router.get("/camera/history")
async def get_camera_history():

    history = []

    cursor = (
        camera_detections_collection
        .find()
        .sort(
            "created_at",
            -1,
        )
        .limit(50)
    )

    async for document in cursor:

        document["_id"] = str(
            document["_id"]
        )

        created_at = document.get(
            "created_at"
        )

        if created_at:

            document["created_at"] = (
                created_at.isoformat()
            )

        filename = document.get(
            "saved_filename"
        )

        if filename:

            document["image_url"] = (
                "/monitoring/camera/image/"
                f"{filename}"
            )

        history.append(
            document
        )

    return {
        "count": len(history),
        "history": history,
    }


# ==========================================
# Get Saved Camera Image
# ==========================================

@router.get(
    "/camera/image/{filename}"
)
async def get_camera_image(
    filename: str,
):

    # Prevent path traversal
    safe_filename = Path(
        filename
    ).name

    image_path = (
        CAMERA_UPLOAD_DIR
        / safe_filename
    )

    if not image_path.exists():

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Camera image not found."
            ),
        )

    return FileResponse(
        path=str(image_path)
    )


# ==========================================
# Combined AI Safety Analysis
# ==========================================

@router.post("/analyze")
async def analyze_safety(

    audio: UploadFile | None = File(
        default=None
    ),

    image: UploadFile | None = File(
        default=None
    ),

    hour: int | None = Form(
        default=None,
        ge=0,
        le=23,
    ),

    alone: bool = Form(
        default=False
    ),

    dark_area: bool = Form(
        default=False
    ),

    # --------------------------------------
    # LOCAL TESTING ONLY
    # --------------------------------------

    test_distress_detected: bool = Form(
        default=False
    ),

    test_fearful_emotion: bool = Form(
        default=False
    ),
):

    """
    Runs available AI detectors and
    combines their signals with the
    risk engine.

    Test flags work only when:

        GUARD_AI_TEST_MODE=true

    in the backend .env file.
    """

    # ======================================
    # Monitoring Check
    # ======================================

    if not monitoring_state["active"]:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Safety monitoring is not active."
            ),
        )

    # ======================================
    # Current Hour
    # ======================================

    if hour is None:
        hour = datetime.now().hour

    # ======================================
    # Results
    # ======================================

    voice_result = None
    emotion_result = None
    speech_result = None
    camera_result = None

    distress_detected = False
    fearful_emotion = False
    emergency_detected = False

    audio_temp_path = None
    image_temp_path = None

    # ======================================
    # Test Mode
    # ======================================

    test_mode = (
        os.getenv(
            "GUARD_AI_TEST_MODE",
            "false",
        ).lower()
        == "true"
    )

    try:

        # ==================================
        # VOICE + EMOTION
        # ==================================

        if (
            audio is not None
            and (
                monitoring_state[
                    "voice_detection"
                ]
                or monitoring_state[
                    "emotion_detection"
                ]
            )
        ):

            filename = (
                audio.filename or ""
            )

            if (
                Path(filename)
                .suffix
                .lower()
                != ".wav"
            ):

                raise HTTPException(
                    status_code=(
                        status.HTTP_400_BAD_REQUEST
                    ),
                    detail=(
                        "Audio must be WAV format."
                    ),
                )

            # Read audio ONCE
            audio_data = await audio.read()

            if not audio_data:

                raise HTTPException(
                    status_code=(
                        status.HTTP_400_BAD_REQUEST
                    ),
                    detail=(
                        "Uploaded audio is empty."
                    ),
                )

            # Save temporary WAV
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            ) as temp_file:

                temp_file.write(
                    audio_data
                )

                audio_temp_path = (
                    temp_file.name
                )

            # ----------------------------------
            # Voice Distress
            # ----------------------------------

            if monitoring_state[
                "voice_detection"
            ]:

                try:

                    detector = (
                        VoiceDistressDetector()
                    )

                    voice_result = (
                        detector.predict(
                            audio_temp_path
                        )
                    )

                    distress_detected = bool(
                        voice_result.get(
                            "distress_detected",
                            False,
                        )
                    )

                except Exception as error:

                    voice_result = {
                        "status":
                            "unavailable",

                        "distress_detected":
                            False,

                        "message":
                            str(error),
                    }

                    distress_detected = False

            # ----------------------------------
            # Emotion
            # ----------------------------------

            if monitoring_state[
                "emotion_detection"
            ]:

                try:

                    detector = (
                        EmotionDetector()
                    )

                    emotion_result = (
                        detector.predict(
                            audio_temp_path
                        )
                    )

                    emotion_name = str(
                        emotion_result.get(
                            "emotion",
                            "",
                        )
                    ).lower()

                    fearful_emotion = (
                        emotion_name
                        in {
                            "fear",
                            "fearful",
                            "scared",
                            "afraid",
                        }
                    )

                except Exception as error:

                    emotion_result = {
                        "status":
                            "unavailable",

                        "emotion":
                            None,

                        "confidence":
                            0,

                        "message":
                            str(error),
                    }

                    fearful_emotion = False

        # ==================================
        # SPEECH EMERGENCY KEYWORD DETECTION
        # ==================================

        if (
            audio_temp_path
            and monitoring_state["active"]
        ):

            try:

                detector = (
                    SpeechKeywordDetector()
                )

                speech_result = (
                    detector.predict(
                        audio_temp_path
                    )
                )

                emergency_detected = bool(
                    speech_result.get(
                        "emergency_detected",
                        False,
                    )
                )

            except Exception as error:

                speech_result = {
                    "status":
                        "unavailable",

                    "emergency_detected":
                        False,

                    "keyword":
                        None,

                    "transcript":
                        "",

                    "message":
                        str(error),
                }

                emergency_detected = False

        # ==================================
        # CAMERA DETECTION
        # ==================================

        if (
            image is not None
            and monitoring_state[
                "camera_detection"
            ]
        ):

            filename = (
                image.filename or ""
            )

            extension = Path(
                filename
            ).suffix.lower()

            if extension not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:

                raise HTTPException(
                    status_code=(
                        status.HTTP_400_BAD_REQUEST
                    ),
                    detail=(
                        "Image must be JPG, "
                        "JPEG or PNG."
                    ),
                )

            image_data = await image.read()

            if not image_data:

                raise HTTPException(
                    status_code=(
                        status.HTTP_400_BAD_REQUEST
                    ),
                    detail=(
                        "Uploaded image is empty."
                    ),
                )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
            ) as temp_file:

                temp_file.write(
                    image_data
                )

                image_temp_path = (
                    temp_file.name
                )

            detector = CameraDetector()

            camera_result = (
                detector.detect(
                    image_temp_path
                )
            )

        # ==================================
        # LOCAL TEST OVERRIDES
        # ==================================

        if test_mode:

            if test_distress_detected:
                distress_detected = True

            if test_fearful_emotion:
                fearful_emotion = True

        # ==================================
        # RISK ENGINE
        # ==================================

        risk_result = calculate_risk(

            hour=hour,

            alone=alone,

            dark_area=dark_area,

            distress_detected=(
                distress_detected
            ),

            fearful_emotion=(
                fearful_emotion
            ),
        )

        # ==================================
        # SOS DECISION
        # ==================================

        sos_required = (
            str(
                risk_result.get(
                    "risk_level",
                    "",
                )
            )
            .lower()
            == "high"
            or emergency_detected
        )

        # ==================================
        # FINAL RESPONSE
        # ==================================

        return {

            "status": "success",

            "monitoring": {

                "active":
                    monitoring_state[
                        "active"
                    ],

                "hour":
                    hour,

                "alone":
                    alone,

                "dark_area":
                    dark_area,
            },

            "ai_results": {

                "voice":
                    voice_result,

                "emotion":
                    emotion_result,

                "speech":
                    speech_result,

                "camera":
                    camera_result,
            },

            "risk":
                risk_result,

            "sos_required":
                sos_required,

            "emergency_keyword_detected":
                emergency_detected,

            "test_mode":
                test_mode,
        }

    except HTTPException:
        raise

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                f"Safety analysis failed: "
                f"{error}"
            ),
        )

    finally:

        # Close uploaded files
        if audio is not None:
            await audio.close()

        if image is not None:
            await image.close()

        # Remove temporary audio
        if (
            audio_temp_path
            and os.path.exists(
                audio_temp_path
            )
        ):

            os.remove(
                audio_temp_path
            )

        # Remove temporary image
        if (
            image_temp_path
            and os.path.exists(
                image_temp_path
            )
        ):

            os.remove(
                image_temp_path
            )