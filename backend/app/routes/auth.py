from fastapi import APIRouter, HTTPException, status

from app.database.database import users_collection
from app.database.models import UserLogin, UserRegister, current_time
from app.utils.security import (
    create_access_token,
    hash_password,
    verify_password,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ==========================================
# Register
# ==========================================

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):

    email = user.email.lower().strip()

    existing_user = await users_collection.find_one(
        {"email": email}
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user_document = {
        "name": user.name.strip(),
        "email": email,
        "password": hash_password(user.password),
        "created_at": current_time(),
    }

    result = await users_collection.insert_one(
        user_document
    )

    return {
        "message": "User registered successfully",
        "user": {
            "id": str(result.inserted_id),
            "name": user.name.strip(),
            "email": email,
        },
    }


# ==========================================
# Login
# ==========================================

@router.post("/login")
async def login(user: UserLogin):

    email = user.email.lower().strip()

    stored_user = await users_collection.find_one(
        {"email": email}
    )

    if not stored_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not verify_password(
        user.password,
        stored_user["password"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(
        user_id=str(stored_user["_id"]),
        email=stored_user["email"],
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(stored_user["_id"]),
            "name": stored_user["name"],
            "email": stored_user["email"],
        },
    }