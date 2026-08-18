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

from app.database.database import contacts_collection
from app.database.models import (
    EmergencyContact,
    current_time,
)
from app.utils.helper import serialize_document
from app.utils.security import decode_access_token


router = APIRouter(
    prefix="/contacts",
    tags=["Emergency Contacts"],
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
    Get logged-in user's ID from JWT.
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
# ADD EMERGENCY CONTACT
# ==========================================

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
)
async def add_contact(
    contact: EmergencyContact,
    user_id: str = Depends(
        get_current_user_id
    ),
):

    contact_document = {
        "user_id": user_id,

        "name": contact.name.strip(),

        "phone": contact.phone.strip(),

        "relation": (
            contact.relation.strip()
            if contact.relation
            else None
        ),

        "created_at": current_time(),
    }

    result = await contacts_collection.insert_one(
        contact_document
    )

    return {
        "message":
            "Emergency contact added successfully",

        "contact": {
            "id":
                str(result.inserted_id),

            "user_id":
                user_id,

            "name":
                contact_document["name"],

            "phone":
                contact_document["phone"],

            "relation":
                contact_document["relation"],
        },
    }


# ==========================================
# GET MY EMERGENCY CONTACTS
# ==========================================

@router.get("/")
async def get_contacts(
    user_id: str = Depends(
        get_current_user_id
    ),
):

    contacts = []

    async for contact in contacts_collection.find(
        {
            "user_id": user_id
        }
    ):

        contacts.append(
            serialize_document(contact)
        )

    return {
        "count":
            len(contacts),

        "contacts":
            contacts,
    }


# ==========================================
# GET SINGLE CONTACT
# ==========================================

@router.get(
    "/{contact_id}"
)
async def get_contact(
    contact_id: str,

    user_id: str = Depends(
        get_current_user_id
    ),
):

    if not ObjectId.is_valid(
        contact_id
    ):

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                "Invalid contact ID",
        )

    contact = await contacts_collection.find_one(
        {
            "_id":
                ObjectId(contact_id),

            "user_id":
                user_id,
        }
    )

    if not contact:

        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                "Contact not found",
        )

    return serialize_document(
        contact
    )


# ==========================================
# DELETE EMERGENCY CONTACT
# ==========================================

@router.delete(
    "/{contact_id}"
)
async def delete_contact(
    contact_id: str,

    user_id: str = Depends(
        get_current_user_id
    ),
):

    if not ObjectId.is_valid(
        contact_id
    ):

        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                "Invalid contact ID",
        )

    result = await contacts_collection.delete_one(
        {
            "_id":
                ObjectId(contact_id),

            "user_id":
                user_id,
        }
    )

    if result.deleted_count == 0:

        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                "Contact not found",
        )

    return {
        "message":
            "Emergency contact deleted successfully"
    }