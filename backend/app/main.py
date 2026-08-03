from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.database import check_database_connection
from app.routes import (
    auth,
    contacts,
    location,
    monitoring,
    risk,
    sos,
)
from app.utils.constants import APP_NAME, APP_VERSION


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs when GuardIA API starts and shuts down.
    """

    database_connected = await check_database_connection()

    if database_connected:
        print("MongoDB connected successfully")
    else:
        print("WARNING: MongoDB is not connected")

    yield

    print("GuardIA API shutting down")


app = FastAPI(
    title=f"{APP_NAME} API",
    description="AI-Powered Women Safety Application",
    version=APP_VERSION,
    lifespan=lifespan,
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Routes
# ==========================================

app.include_router(auth.router)
app.include_router(contacts.router)
app.include_router(location.router)
app.include_router(sos.router)
app.include_router(risk.router)
app.include_router(monitoring.router)


# ==========================================
# Root
# ==========================================

@app.get("/")
async def root():
    return {
        "project": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "message": "GuardIA API is running successfully",
    }


# ==========================================
# Health
# ==========================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": APP_NAME,
    }