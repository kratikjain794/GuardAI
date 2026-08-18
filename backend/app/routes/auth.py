from fastapi import APIRouter, HTTPException, status

from app.database.database import users_collection
from app.database.models import (
    UserLogin,
    UserRegister,
    current_time,
)
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

@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
)
async def register(user: UserRegister):

    # --------------------------------------
    # Normalize data
    # --------------------------------------

    name = user.name.strip()
    phone = user.phone.strip()
    email = user.email.lower().strip()

    # --------------------------------------
    # Check duplicate email
    # --------------------------------------

    existing_user = await users_collection.find_one(
        {
            "email": email
        }
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # --------------------------------------
    # Check duplicate phone
    # --------------------------------------

    existing_phone = await users_collection.find_one(
        {
            "phone": phone
        }
    )

    if existing_phone:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Mobile number already registered",
        )

    # --------------------------------------
    # Create user document
    # --------------------------------------

    user_document = {
        "name": name,
        "phone": phone,
        "email": email,
        "password": hash_password(
            user.password
        ),
        "created_at": current_time(),
    }

    # --------------------------------------
    # Insert user
    # --------------------------------------

    result = await users_collection.insert_one(
        user_document
    )

    # --------------------------------------
    # Response
    # --------------------------------------

    return {
        "message": "User registered successfully",
        "user": {
            "id": str(
                result.inserted_id
            ),
            "name": name,
            "phone": phone,
            "email": email,
        },
    }


# ==========================================
# Login
# ==========================================

@router.post("/login")
async def login(user: UserLogin):

    email = user.email.lower().strip()

    # --------------------------------------
    # Find user
    # --------------------------------------

    stored_user = await users_collection.find_one(
        {
            "email": email
        }
    )

    if not stored_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # --------------------------------------
    # Verify password
    # --------------------------------------

    if not verify_password(
        user.password,
        stored_user["password"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # --------------------------------------
    # Create JWT
    # --------------------------------------

    access_token = create_access_token(
        user_id=str(
            stored_user["_id"]
        ),
        email=stored_user["email"],
    )

    # --------------------------------------
    # Existing users compatibility
    # --------------------------------------

    phone = stored_user.get("phone")

    # --------------------------------------
    # Response
    # --------------------------------------

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(
                stored_user["_id"]
            ),
            "name": stored_user.get(
                "name",
                "",
            ),
            "phone": phone,
            "email": stored_user["email"],
        },
    }