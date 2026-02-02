from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
import sys

from swarmcraft.api.routes import router
from swarmcraft.api.websocket import websocket_manager, handle_websocket_message
from swarmcraft.database.redis_client import redis_client, get_redis

load_dotenv()

# Configure Loguru for clear, colored, and timestamped logging
logger.remove()
logger.add(
    sys.stderr,
    # format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan management"""
    # Startup
    logger.info("Starting SwarmCraft API...")
    try:
        await redis_client.connect()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down SwarmCraft API...")
    await redis_client.disconnect()
    logger.info("Redis connection closed")


app = FastAPI(
    title="SwarmCraft API",
    description="Interactive swarm intelligence for experiential learning",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS for frontend
# Get allowed origins from environment variable, default to "*" if not set
cors_origins_str = os.getenv("CORS_ORIGINS", "*")
origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all dependencies"""
    health_status = {"status": "healthy", "services": {}}

    # Test Redis connection
    try:
        redis_conn = await get_redis()
        await redis_conn.ping()
        health_status["services"]["redis"] = {"status": "connected"}
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["services"]["redis"] = {"status": "error", "message": str(e)}

    # Test import capabilities
    try:
        from swarmcraft.core.loss_functions import create_landscape

        from swarmcraft.config import DEFAULT_GRID_SIZE

        _ = create_landscape(
            "rastrigin", grid_size=DEFAULT_GRID_SIZE, A=10.0, dimensions=2
        )
        health_status["services"]["landscapes"] = {"status": "functional"}
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["services"]["landscapes"] = {"status": "error", "message": str(e)}

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


# Include API routes
app.include_router(router, prefix="/api")


@app.websocket("/ws/{session_id}/{participant_id}")
async def websocket_endpoint(
    websocket: WebSocket, session_id: str, participant_id: str
):
    await websocket_manager.connect(websocket, session_id, participant_id)
    await handle_websocket_message(websocket, session_id, participant_id)
