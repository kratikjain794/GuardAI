from fastapi import APIRouter, status

from app.database.database import locations_collection
from app.database.models import LocationData, current_time


router = APIRouter(
    prefix="/location",
    tags=["Location"],
)


# ==========================================
# Update / Store Current Location
# ==========================================

@router.post("/", status_code=status.HTTP_201_CREATED)
async def update_location(location: LocationData):

    location_document = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timestamp": current_time(),
    }

    result = await locations_collection.insert_one(
        location_document
    )

    return {
        "message": "Location updated successfully",
        "location_id": str(result.inserted_id),
        "location": {
            "latitude": location.latitude,
            "longitude": location.longitude,
        },
    }


# ==========================================
# Get Latest Location
# ==========================================

@router.get("/latest")
async def get_latest_location():

    location = await locations_collection.find_one(
        sort=[("timestamp", -1)]
    )

    if not location:
        return {
            "message": "No location available",
            "location": None,
        }

    return {
        "location": {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "timestamp": location["timestamp"],
        }
    }