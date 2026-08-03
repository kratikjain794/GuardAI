from bson import ObjectId
from fastapi import APIRouter, HTTPException, status

from app.database.database import sos_collection
from app.database.models import SOSRequest, current_time
from app.utils.helper import serialize_document


router = APIRouter(
    prefix="/sos",
    tags=["SOS"],
)


# ==========================================
# Trigger SOS
# ==========================================

@router.post("/", status_code=status.HTTP_201_CREATED)
async def trigger_sos(request: SOSRequest):

    sos_document = {
        "latitude": request.latitude,
        "longitude": request.longitude,
        "message": request.message,
        "status": "active",
        "created_at": current_time(),
    }

    result = await sos_collection.insert_one(
        sos_document
    )

    return {
        "message": "SOS activated successfully",
        "sos_id": str(result.inserted_id),
        "status": "active",
        "location": {
            "latitude": request.latitude,
            "longitude": request.longitude,
        },
    }


# ==========================================
# Get SOS Event
# ==========================================

@router.get("/{sos_id}")
async def get_sos(sos_id: str):

    if not ObjectId.is_valid(sos_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid SOS ID",
        )

    sos = await sos_collection.find_one(
        {"_id": ObjectId(sos_id)}
    )

    if not sos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SOS event not found",
        )

    return serialize_document(sos)


# ==========================================
# Cancel / Resolve SOS
# ==========================================

@router.patch("/{sos_id}/resolve")
async def resolve_sos(sos_id: str):

    if not ObjectId.is_valid(sos_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid SOS ID",
        )

    result = await sos_collection.update_one(
        {"_id": ObjectId(sos_id)},
        {
            "$set": {
                "status": "resolved",
                "resolved_at": current_time(),
            }
        },
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SOS event not found",
        )

    return {
        "message": "SOS resolved successfully",
        "sos_id": sos_id,
        "status": "resolved",
    }