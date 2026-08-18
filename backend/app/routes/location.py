from bson import ObjectId
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from app.database.database import locations_collection
from app.database.models import (
    LocationData,
    current_time,
)
from app.utils.security import decode_access_token


router = APIRouter(
    prefix="/location",
    tags=["Location"],
)


# ==========================================
# AUTHENTICATION
# ==========================================

bearer_scheme = HTTPBearer(
    auto_error=True
)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
):
    """
    Get logged-in user ID from JWT.
    """

    token = credentials.credentials

    payload = decode_access_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )

    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
        )

    if not ObjectId.is_valid(user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID",
        )

    return user_id


# ==========================================
# UPDATE / STORE CURRENT LOCATION
# ==========================================

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
)
async def update_location(
    location: LocationData,
    user_id: str = Depends(
        get_current_user_id
    ),
):

    location_document = {
        "user_id": user_id,

        "latitude": location.latitude,

        "longitude": location.longitude,

        "timestamp": current_time(),
    }

    result = await locations_collection.insert_one(
        location_document
    )

    return {
        "message":
            "Location updated successfully",

        "location_id":
            str(result.inserted_id),

        "user_id":
            user_id,

        "location": {
            "latitude":
                location.latitude,

            "longitude":
                location.longitude,
        },
    }


# ==========================================
# GET LATEST LOCATION
# ==========================================

@router.get(
    "/latest"
)
async def get_latest_location(
    user_id: str = Depends(
        get_current_user_id
    ),
):

    location = await locations_collection.find_one(
        {
            "user_id": user_id
        },

        sort=[
            ("timestamp", -1)
        ],
    )

    if not location:

        return {
            "message":
                "No location available",

            "location":
                None,
        }

    return {
        "user_id":
            user_id,

        "location": {
            "latitude":
                location["latitude"],

            "longitude":
                location["longitude"],

            "timestamp":
                location["timestamp"],
        },
    }