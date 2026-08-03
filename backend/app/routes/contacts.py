from bson import ObjectId
from fastapi import APIRouter, HTTPException, status

from app.database.database import contacts_collection
from app.database.models import EmergencyContact, current_time
from app.utils.helper import serialize_document


router = APIRouter(
    prefix="/contacts",
    tags=["Emergency Contacts"],
)


# ==========================================
# Add Emergency Contact
# ==========================================

@router.post("/", status_code=status.HTTP_201_CREATED)
async def add_contact(contact: EmergencyContact):

    contact_document = {
        **contact.model_dump(),
        "created_at": current_time(),
    }

    result = await contacts_collection.insert_one(
        contact_document
    )

    return {
        "message": "Emergency contact added successfully",
        "contact": {
            "id": str(result.inserted_id),
            **contact.model_dump(),
        },
    }


# ==========================================
# Get All Emergency Contacts
# ==========================================

@router.get("/")
async def get_contacts():

    contacts = []

    async for contact in contacts_collection.find():
        contacts.append(
            serialize_document(contact)
        )

    return {
        "count": len(contacts),
        "contacts": contacts,
    }


# ==========================================
# Get Single Contact
# ==========================================

@router.get("/{contact_id}")
async def get_contact(contact_id: str):

    if not ObjectId.is_valid(contact_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid contact ID",
        )

    contact = await contacts_collection.find_one(
        {"_id": ObjectId(contact_id)}
    )

    if not contact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact not found",
        )

    return serialize_document(contact)


# ==========================================
# Delete Emergency Contact
# ==========================================

@router.delete("/{contact_id}")
async def delete_contact(contact_id: str):

    if not ObjectId.is_valid(contact_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid contact ID",
        )

    result = await contacts_collection.delete_one(
        {"_id": ObjectId(contact_id)}
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact not found",
        )

    return {
        "message": "Emergency contact deleted successfully"
    }