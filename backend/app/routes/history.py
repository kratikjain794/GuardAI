from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from app.database.database import sos_collection
from app.utils.security import decode_access_token


router = APIRouter(
    prefix="/history",
    tags=["History"],
)

bearer_scheme = HTTPBearer(
    auto_error=True
)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
):
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

    return user_id


@router.get("/")
async def get_history(
    user_id: str = Depends(
        get_current_user_id
    ),
):
    history = []

    async for sos in sos_collection.find(
        {
            "user_id": user_id
        }
    ).sort(
        "created_at",
        -1
    ):
        history.append(
            {
                "id": str(sos["_id"]),
                "type": "SOS",
                "status": sos.get(
                    "status",
                    "unknown"
                ),
                "message": sos.get(
                    "message",
                    ""
                ),
                "latitude": sos.get(
                    "latitude"
                ),
                "longitude": sos.get(
                    "longitude"
                ),
                "created_at": sos.get(
                    "created_at"
                ),
                "resolved_at": sos.get(
                    "resolved_at"
                ),
            }
        )

    return {
        "count": len(history),
        "history": history,
    }

