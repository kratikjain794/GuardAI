import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017"
)

DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    "guardia"
)

# MongoDB client
client = AsyncIOMotorClient(MONGO_URI)

# GuardAI database
database = client[DATABASE_NAME]

# Collections
users_collection = database["users"]
contacts_collection = database["contacts"]
locations_collection = database["locations"]
sos_collection = database["sos_alerts"]

# Camera detection history
camera_detections_collection = database[
    "camera_detections"
]


async def check_database_connection() -> bool:
    """Check whether MongoDB is reachable."""

    try:
        await client.admin.command("ping")
        return True

    except Exception as error:
        print(
            f"MongoDB connection failed: {error}"
        )
        return False