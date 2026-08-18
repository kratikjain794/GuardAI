import os

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

from twilio.rest import Client

from app.database.database import (
    sos_collection,
    contacts_collection,
    users_collection,
)

from app.database.models import (
    SOSRequest,
    current_time,
)

from app.utils.helper import serialize_document
from app.utils.security import decode_access_token


# ==========================================
# ROUTER
# ==========================================

router = APIRouter(
    prefix="/sos",
    tags=["SOS"],
)


# ==========================================
# AUTHENTICATION
# ==========================================

bearer_scheme = HTTPBearer(
    auto_error=True
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
):
    """
    Get currently logged-in user from JWT.
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
            detail="Invalid user ID in access token",
        )

    user = await users_collection.find_one(
        {
            "_id": ObjectId(user_id)
        }
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


# ==========================================
# TWILIO CONFIGURATION
# ==========================================

TWILIO_ACCOUNT_SID = os.getenv(
    "TWILIO_ACCOUNT_SID"
)

TWILIO_API_KEY_SID = os.getenv(
    "TWILIO_API_KEY_SID"
)

TWILIO_API_KEY_SECRET = os.getenv(
    "TWILIO_API_KEY_SECRET"
)

TWILIO_PHONE_NUMBER = os.getenv(
    "TWILIO_PHONE_NUMBER"
)


# ==========================================
# TWILIO SMS
# ==========================================

def send_sms(
    to_number: str,
):
    """
    Send SMS using Twilio.

    NOTE:
    Current Twilio trial account only accepts
    predefined SMS templates.

    Therefore the trial-compatible template
    is used here.
    """

    if not all(
        [
            TWILIO_ACCOUNT_SID,
            TWILIO_API_KEY_SID,
            TWILIO_API_KEY_SECRET,
            TWILIO_PHONE_NUMBER,
        ]
    ):
        return {
            "status": "failed",
            "message": "Twilio configuration is missing.",
        }

    if not to_number:
        return {
            "status": "failed",
            "message": "Emergency contact phone number is missing.",
        }

    try:
        client = Client(
            TWILIO_API_KEY_SID,
            TWILIO_API_KEY_SECRET,
            account_sid=TWILIO_ACCOUNT_SID,
        )

        # Twilio trial-compatible predefined template
        message = client.messages.create(
            body="sms_appointment_reminders",
            from_=TWILIO_PHONE_NUMBER,
            to=to_number,
        )

        return {
            "status": "sent",
            "sid": message.sid,
            "delivery": message.status,
        }

    except Exception as error:
        return {
            "status": "failed",
            "message": str(error),
        }


# ==========================================
# TRIGGER SOS
# ==========================================

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
)
async def trigger_sos(
    request: SOSRequest,
    current_user=Depends(get_current_user),
):

    # --------------------------------------
    # 1. Logged-in user information
    # --------------------------------------

    user_id = str(
        current_user["_id"]
    )

    user_name = current_user.get(
        "name",
        "",
    )

    user_phone = current_user.get(
        "phone",
        "",
    )

    user_email = current_user.get(
        "email",
        "",
    )

    # --------------------------------------
    # 2. Create SOS document
    # --------------------------------------

    sos_document = {
        "user_id": user_id,

        "user_name": user_name,

        "user_phone": user_phone,

        "user_email": user_email,

        "latitude": request.latitude,

        "longitude": request.longitude,

        "message": request.message,

        "status": "active",

        "created_at": current_time(),
    }

    result = await sos_collection.insert_one(
        sos_document
    )

    sos_id = str(
        result.inserted_id
    )

    # --------------------------------------
    # 3. Get ONLY current user's contacts
    # --------------------------------------

    contacts = []

    async for contact in contacts_collection.find(
        {
            "user_id": user_id
        }
    ):

        contacts.append(
            {
                "id": str(
                    contact["_id"]
                ),

                "name": contact.get(
                    "name",
                    "",
                ),

                "phone": contact.get(
                    "phone",
                    "",
                ),

                "relation": contact.get(
                    "relation"
                ),
            }
        )

    # --------------------------------------
    # 4. Google Maps Location
    # --------------------------------------

    maps_url = (
        "https://www.google.com/maps?q="
        f"{request.latitude},"
        f"{request.longitude}"
    )

    # --------------------------------------
    # 5. Prepare Alert Message
    # --------------------------------------

    alert_message = (
        "GUARD AI EMERGENCY ALERT\n\n"

        f"{user_name} may be in a "
        "high-risk safety situation.\n\n"

        f"User Mobile: {user_phone}\n\n"

        f"SOS ID: {sos_id}\n\n"

        "Location:\n"
        f"{maps_url}\n\n"

        "Coordinates: "
        f"{request.latitude}, "
        f"{request.longitude}\n\n"

        "Message: "
        f"{request.message}\n\n"

        "Please check the person's "
        "location immediately."
    )

    # --------------------------------------
    # 6. Send SMS to emergency contacts
    # --------------------------------------

    contact_alerts = []

    sms_results = []

    for contact in contacts:

        phone = contact.get(
            "phone",
            ""
        )

        # Send SMS
        sms_result = send_sms(
            phone
        )

        sms_results.append(
            {
                "contact_id": contact["id"],
                "name": contact["name"],
                "phone": phone,
                "status": sms_result.get(
                    "status"
                ),
                "sid": sms_result.get(
                    "sid"
                ),
                "delivery": sms_result.get(
                    "delivery"
                ),
                "error": sms_result.get(
                    "message"
                ),
            }
        )

        # Existing alert information
        contact_alerts.append(
            {
                "contact_id":
                    contact["id"],

                "name":
                    contact["name"],

                "phone":
                    contact["phone"],

                "relation":
                    contact["relation"],

                "alert_status":
                    (
                        "sent"
                        if sms_result.get(
                            "status"
                        ) == "sent"
                        else "failed"
                    ),

                "message":
                    alert_message,

                "sms_sid":
                    sms_result.get(
                        "sid"
                    ),

                "sms_delivery":
                    sms_result.get(
                        "delivery"
                    ),

                "sms_error":
                    sms_result.get(
                        "message"
                    ),
            }
        )

    # --------------------------------------
    # 7. Calculate SMS status
    # --------------------------------------

    successful_sms = sum(
        1
        for result in sms_results
        if result.get("status") == "sent"
    )

    failed_sms = sum(
        1
        for result in sms_results
        if result.get("status") == "failed"
    )

    if successful_sms > 0:
        notification_status = "sent"
        notification_message = (
            "Emergency contact SMS sent successfully."
        )
        delivery_status = "twilio"
    elif len(contacts) == 0:
        notification_status = "no_contacts"
        notification_message = (
            "No emergency contacts found."
        )
        delivery_status = "not_sent"
    else:
        notification_status = "failed"
        notification_message = (
            "Failed to send emergency contact SMS."
        )
        delivery_status = "twilio_error"

    # --------------------------------------
    # 8. Return SOS Information
    # --------------------------------------

    return {
        "message":
            "SOS activated successfully",

        "sos_id":
            sos_id,

        "status":
            "active",

        "user": {
            "user_id":
                user_id,

            "name":
                user_name,

            "phone":
                user_phone,

            "email":
                user_email,
        },

        "location": {
            "latitude":
                request.latitude,

            "longitude":
                request.longitude,

            "maps_url":
                maps_url,
        },

        "emergency_contacts": {
            "count":
                len(contacts),

            "contacts":
                contact_alerts,
        },

        "notification": {
            "status":
                notification_status,

            "message":
                notification_message,

            "delivery":
                delivery_status,

            "sent_count":
                successful_sms,

            "failed_count":
                failed_sms,

            "sms_results":
                sms_results,
        },
    }


# ==========================================
# GET SOS EVENT
# ==========================================

@router.get(
    "/{sos_id}"
)
async def get_sos(
    sos_id: str,
    current_user=Depends(
        get_current_user
    ),
):

    # --------------------------------------
    # Validate ObjectId
    # --------------------------------------

    if not ObjectId.is_valid(
        sos_id
    ):

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                "Invalid SOS ID",
        )

    # --------------------------------------
    # Find SOS
    # --------------------------------------

    sos = await sos_collection.find_one(
        {
            "_id":
                ObjectId(sos_id)
        }
    )

    if not sos:

        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                "SOS event not found",
        )

    # --------------------------------------
    # Security:
    # Only owner can view SOS
    # --------------------------------------

    current_user_id = str(
        current_user["_id"]
    )

    if sos.get(
        "user_id"
    ) != current_user_id:

        raise HTTPException(
            status_code=
                status.HTTP_403_FORBIDDEN,

            detail=
                "You are not allowed to access this SOS",
        )

    # --------------------------------------
    # Serialize
    # --------------------------------------

    return serialize_document(
        sos
    )


# ==========================================
# RESOLVE SOS
# ==========================================

@router.patch(
    "/{sos_id}/resolve"
)
async def resolve_sos(
    sos_id: str,
    current_user=Depends(
        get_current_user
    ),
):

    # --------------------------------------
    # Validate ObjectId
    # --------------------------------------

    if not ObjectId.is_valid(
        sos_id
    ):

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                "Invalid SOS ID",
        )

    # --------------------------------------
    # Find SOS
    # --------------------------------------

    sos = await sos_collection.find_one(
        {
            "_id":
                ObjectId(sos_id)
        }
    )

    if not sos:

        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                "SOS event not found",
        )

    # --------------------------------------
    # Security:
    # Only owner can resolve SOS
    # --------------------------------------

    current_user_id = str(
        current_user["_id"]
    )

    if sos.get(
        "user_id"
    ) != current_user_id:

        raise HTTPException(
            status_code=
                status.HTTP_403_FORBIDDEN,

            detail=
                "You are not allowed to resolve this SOS",
        )

    # --------------------------------------
    # Update SOS
    # --------------------------------------

    result = await sos_collection.update_one(
        {
            "_id":
                ObjectId(sos_id)
        },

        {
            "$set": {
                "status":
                    "resolved",

                "resolved_at":
                    current_time(),
            }
        },
    )

    if result.matched_count == 0:

        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                "SOS event not found",
        )

    return {
        "message":
            "SOS resolved successfully",

        "sos_id":
            sos_id,

        "status":
            "resolved",
    }