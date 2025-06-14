from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional, List
import os
from datetime import datetime

from swarmcraft.database.redis_client import get_redis, set_json, get_json
from swarmcraft.models.session import (
    GameSession,
    SessionConfig,
    Participant,
    SessionStatus,
)
from swarmcraft.utils.name_generator import (
    generate_participant_name,
    generate_session_code,
)
from swarmcraft.core.loss_functions import create_landscape
from swarmcraft.api.websocket import websocket_manager

router = APIRouter()


async def verify_admin_key(x_admin_key: Optional[str] = Header(None)):
    """Verify admin secret key"""
    expected_key = os.getenv("SWARM_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Admin key not configured")
    if x_admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


@router.post("/admin/create-session")
async def create_session(
    config: SessionConfig,
    redis_conn=Depends(get_redis),
    _: bool = Depends(verify_admin_key),
):
    """Create a new game session (admin only)"""
    session_id = generate_session_code()
    session_code = generate_session_code()

    session = GameSession(
        id=session_id,
        code=session_code,
        admin_id="admin",  # Could be more sophisticated
        config=config,
        created_at=datetime.now(),
    )

    # Store in Redis
    await set_json(
        f"session:{session_id}", session.model_dump(), redis_conn, expire=86400
    )  # 24h TTL
    await redis_conn.setex(f"session_code:{session_code}", 86400, session_id)

    return {
        "session_id": session_id,
        "session_code": session_code,
        "message": f"Session created! Share code '{session_code}' with participants",
    }


@router.post("/join/{session_code}")
async def join_session(session_code: str, redis_conn=Depends(get_redis)):
    """Join a session using session code"""
    # Get session ID from code
    session_id = await redis_conn.get(f"session_code:{session_code}")
    if not session_id:
        raise HTTPException(status_code=404, detail="Invalid session code")

    session_id = session_id.decode()

    # Get session data
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)

    # Check if session is joinable
    if session.status not in [SessionStatus.WAITING, SessionStatus.ACTIVE]:
        raise HTTPException(
            status_code=400, detail="Session not accepting participants"
        )

    if len(session.participants) >= session.config.max_participants:
        raise HTTPException(status_code=400, detail="Session full")

    # Create participant
    participant_id = f"p_{len(session.participants) + 1}"
    participant = Participant(
        id=participant_id, name=generate_participant_name(), joined_at=datetime.now()
    )

    # Add to session
    session.participants.append(participant)

    # Save updated session
    await set_json(
        f"session:{session_id}", session.model_dump(), redis_conn, expire=86400
    )

    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "participant_name": participant.name,
        "message": f"Welcome, {participant.name}!",
    }


@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str, redis_conn=Depends(get_redis)):
    """Get current session status"""
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)

    return {
        "status": session.status.value,
        "participants": len(session.participants),
        "max_participants": session.config.max_participants,
        "landscape_type": session.config.landscape_type,
        "grid_size": session.config.grid_size,
        "iteration": session.swarm_iteration,
    }


@router.post("/session/{session_id}/move")
async def make_move(
    session_id: str,
    participant_id: str,
    position: List[int],  # [i, j] grid coordinates
    redis_conn=Depends(get_redis),
):
    """Participant makes a move"""
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)

    # Find participant
    participant = None
    for p in session.participants:
        if p.id == participant_id:
            participant = p
            break

    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    # Validate position
    grid_size = session.config.grid_size
    if not (0 <= position[0] < grid_size and 0 <= position[1] < grid_size):
        raise HTTPException(status_code=400, detail="Position out of bounds")

    # Update participant position
    participant.position = position

    # Calculate fitness using discrete landscape
    landscape = create_landscape(
        session.config.landscape_type, **session.config.landscape_params
    )

    # Convert grid coordinates to continuous space and evaluate
    bounds = landscape.metadata.recommended_bounds
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Grid cell center coordinates
    x = x_min + (position[1] + 0.5) * (x_max - x_min) / grid_size
    y = y_min + (position[0] + 0.5) * (y_max - y_min) / grid_size

    fitness = landscape.evaluate([x, y])
    participant.fitness = fitness

    # Save updated session
    await set_json(
        f"session:{session_id}", session.model_dump(), redis_conn, expire=86400
    )

    # Get feedback
    color = landscape.get_fitness_color(fitness)
    frequency = landscape.get_fitness_audio_frequency(fitness)
    description = landscape.describe_position([x, y])

    return {
        "position": position,
        "fitness": fitness,
        "color": color,
        "frequency": frequency,
        "description": description,
    }


@router.post("/admin/session/{session_id}/step")
async def trigger_swarm_step(session_id: str, _: bool = Depends(verify_admin_key)):
    """Manually trigger a swarm optimization step"""
    await websocket_manager.run_swarm_step(session_id)
    return {"message": "Swarm step executed"}


@router.post("/admin/session/{session_id}/start")
async def start_session(
    session_id: str, redis_conn=Depends(get_redis), _: bool = Depends(verify_admin_key)
):
    """Start the session"""
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)
    session.status = SessionStatus.ACTIVE
    session.started_at = datetime.now()

    await set_json(
        f"session:{session_id}", session.model_dump(), redis_conn, expire=86400
    )

    # Notify all participants
    await websocket_manager.broadcast_to_session(
        {
            "type": "session_started",
            "message": "The swarm optimization has begun!",
            "timestamp": datetime.now().isoformat(),
        },
        session_id,
    )

    return {"message": "Session started"}
