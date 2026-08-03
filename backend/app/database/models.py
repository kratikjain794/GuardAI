from datetime import datetime, timezone

from pydantic import BaseModel, EmailStr, Field


# -------------------------
# User Models
# -------------------------

class UserRegister(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# -------------------------
# Emergency Contact
# -------------------------

class EmergencyContact(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    phone: str = Field(min_length=7, max_length=20)
    relation: str | None = None


# -------------------------
# Location
# -------------------------

class LocationData(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


# -------------------------
# SOS
# -------------------------

class SOSRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    message: str = (
        "Emergency! I need help. "
        "Please check my location."
    )


# -------------------------
# Utility
# -------------------------

def current_time() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)