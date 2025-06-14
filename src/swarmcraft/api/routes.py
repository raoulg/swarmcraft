from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional
import os
from datetime import datetime
import numpy as np

from swarmcraft.database.redis_client import get_redis, set_json, get_json
from swarmcraft.models.session import (
    GameSession,
    SessionConfig,
    Participant,
    SessionStatus,
    MoveData,
)
from swarmcraft.utils.name_generator import (
    generate_participant_name,
    generate_session_code,
)
from swarmcraft.core.loss_functions import create_landscape
from swarmcraft.core.swarm_base import SwarmState
from swarmcraft.core.pso import PSO
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
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
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
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
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
    move_data: MoveData,
    redis_conn=Depends(get_redis),
):
    """
    Triggers a rule-based move for a participant's particle using the
    persistent SwarmState from Redis.
    """
    # 1. LOAD both the GameSession and the full SwarmState from Redis.
    session_data = await get_json(f"session:{session_id}", redis_conn)
    swarm_state_data = await get_json(f"swarm_state:{session_id}", redis_conn)

    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    if not swarm_state_data:
        raise HTTPException(
            status_code=404,
            detail="Swarm state not found. The session may not have been started yet.",
        )

    session = GameSession(**session_data)
    swarm_state = SwarmState(**swarm_state_data)

    if session.status != SessionStatus.ACTIVE:
        raise HTTPException(
            status_code=400, detail="Moves can only be made in an active session."
        )

    # 2. HYDRATE a temporary PSO "calculator" with the loaded state.
    landscape = create_landscape(
        session.config.landscape_type, **session.config.landscape_params
    )
    pso = PSO(
        dimensions=2,
        bounds=landscape.metadata.recommended_bounds,
        loss_function=landscape.evaluate,
        population_size=len(session.participants),
    )
    # This is the crucial step: restoring the full state of the swarm.
    pso.swarm_state = swarm_state

    # 3. Find the correct internal particle ID for the participant making the move.
    participant_index = next(
        (
            i
            for i, p in enumerate(session.participants)
            if p.id == move_data.participant_id
        ),
        -1,
    )
    if participant_index == -1:
        raise HTTPException(status_code=404, detail="Participant not found")
    pso_particle_id = f"particle_{participant_index}"

    # 4. MUTATE the state by calculating the particle's next move.
    suggested_cont_pos = pso.suggest_next_position(pso_particle_id)
    if suggested_cont_pos is None:
        raise HTTPException(
            status_code=500, detail="Could not calculate next position."
        )

    # 5. Update the state of the specific particle within the swarm_state object.
    particle_to_update = pso.get_particle_by_id(pso_particle_id)
    new_fitness = landscape.evaluate(np.array(suggested_cont_pos))
    if particle_to_update:
        particle_to_update.update_position(np.array(suggested_cont_pos))
        particle_to_update.fitness = new_fitness

    # 6. Convert the new continuous position back to a discrete grid position.
    bounds = landscape.metadata.recommended_bounds
    grid_size = session.config.grid_size
    new_x, new_y = suggested_cont_pos[0], suggested_cont_pos[1]
    col = int((new_x - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) * grid_size)
    row = int((new_y - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) * grid_size)
    new_grid_pos = [max(0, min(grid_size - 1, row)), max(0, min(grid_size - 1, col))]

    # 7. Update the participant's info in the GameSession object.
    session.participants[participant_index].position = new_grid_pos
    session.participants[participant_index].fitness = new_fitness

    # 8. SAVE both updated states back to Redis.
    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )
    await set_json(
        f"swarm_state:{session_id}",
        pso.swarm_state.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )

    # 9. Return feedback to the user.
    return {
        "position": new_grid_pos,
        "fitness": new_fitness,
        "color": landscape.get_fitness_color(new_fitness),
        "frequency": landscape.get_fitness_audio_frequency(new_fitness),
        "description": landscape.describe_position(np.array(suggested_cont_pos)),
    }


@router.post("/admin/session/{session_id}/step")
async def trigger_swarm_step(
    session_id: str, redis_conn=Depends(get_redis), _: bool = Depends(verify_admin_key)
):
    """
    Executes a full, synchronous step of the PSO algorithm, updating all
    particles at once.
    """
    # 1. LOAD both the GameSession and the full SwarmState from Redis
    session_data = await get_json(f"session:{session_id}", redis_conn)
    swarm_state_data = await get_json(f"swarm_state:{session_id}", redis_conn)

    if not session_data or not swarm_state_data:
        raise HTTPException(status_code=404, detail="Session or Swarm state not found.")

    session = GameSession(**session_data)
    swarm_state = SwarmState(**swarm_state_data)

    if session.status != SessionStatus.ACTIVE:
        return {"message": "Swarm step skipped: session is not active."}

    # 2. HYDRATE a temporary PSO instance with the loaded state
    landscape = create_landscape(
        session.config.landscape_type, **session.config.landscape_params
    )
    pso = PSO(
        dimensions=2,
        bounds=landscape.metadata.recommended_bounds,
        loss_function=landscape.evaluate,
        population_size=len(session.participants),
    )
    pso.swarm_state = swarm_state

    # 3. MUTATE the state by executing one full step of the algorithm
    #    This updates all particles in pso.swarm_state in-memory.
    pso.step()

    # 4. SYNC the GameSession with the new state of all particles
    session.participants = pso.sync_participants_from_swarm(
        participants=session.participants, grid_size=session.config.grid_size
    )
    session.swarm_iteration = pso.swarm_state.iteration

    # 5. SAVE both updated states back to Redis
    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )
    await set_json(
        f"swarm_state:{session_id}",
        pso.swarm_state.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )

    await websocket_manager.send_session_state(session_id)

    # Note: After this call, you would likely want to broadcast the full
    # new state to all participants via WebSockets so their phones update.
    # We can add that logic to websocket.py later.

    return {"message": f"Swarm step {session.swarm_iteration} executed successfully."}


@router.post("/admin/session/{session_id}/start")
async def start_session(
    session_id: str, redis_conn=Depends(get_redis), _: bool = Depends(verify_admin_key)
):
    """
    Starts the session, initializes the swarm with random positions,
    and assigns those positions to the participants.
    """
    # 1. LOAD the GameSession state from Redis
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)
    if not session.participants:
        raise HTTPException(
            status_code=400, detail="Cannot start a session with no participants."
        )

    session.status = SessionStatus.ACTIVE
    session.started_at = datetime.now()

    # 2. CREATE the PSO instance (our temporary "calculator").
    #    It will initialize its own internal SwarmState with random positions.
    landscape = create_landscape(
        session.config.landscape_type, **session.config.landscape_params
    )
    pso = PSO(
        dimensions=2,
        bounds=landscape.metadata.recommended_bounds,
        loss_function=landscape.evaluate,
        population_size=len(session.participants),
    )

    # 3. SYNC the game state with the newly created swarm state.
    #    The PSO object updates the list of participants with their assigned positions.
    session.participants = pso.sync_participants_from_swarm(
        participants=session.participants, grid_size=session.config.grid_size
    )

    # 4. SAVE the updated states back to Redis (our "source of truth").
    #    We save both the GameSession (with updated participant positions)
    #    and the full SwarmState (with velocities, personal bests, etc.).
    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )
    await set_json(
        f"swarm_state:{session_id}",
        pso.swarm_state.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )

    # 5. NOTIFY clients that the game is on!
    await websocket_manager.broadcast_to_session(
        {
            "type": "session_started",
            "message": "The swarm optimization has begun! You have been assigned a starting position.",
            "timestamp": datetime.now().isoformat(),
        },
        session_id,
    )

    return {"message": "Session started and swarm state initialized."}


@router.get("/admin/sessions")
async def list_sessions(
    redis_conn=Depends(get_redis), _: bool = Depends(verify_admin_key)
):
    """Lists all active game sessions."""
    session_keys = [key async for key in redis_conn.scan_iter("session:*")]
    sessions_summary = []
    for key in session_keys:
        session_data = await get_json(key, redis_conn)
        if session_data:
            sessions_summary.append(
                {
                    "session_id": session_data.get("id"),
                    "session_code": session_data.get("code"),
                    "status": session_data.get("status"),
                    "participant_count": len(session_data.get("participants", [])),
                    "created_at": session_data.get("created_at"),
                }
            )
    return sessions_summary


@router.delete("/admin/session/{session_id}")
async def close_session(
    session_id: str,
    redis_conn=Depends(get_redis),
    _: bool = Depends(verify_admin_key),
):
    """Closes a specific session and removes all associated data."""
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    session_code = session_data.get("code")

    # Delete all related keys from Redis
    await redis_conn.delete(f"session:{session_id}")
    await redis_conn.delete(f"swarm_state:{session_id}")
    if session_code:
        await redis_conn.delete(f"session_code:{session_code}")

    # Disconnect any connected WebSocket clients for this session
    await websocket_manager.broadcast_to_session(
        {
            "type": "session_closed",
            "message": "The session has been closed by the admin.",
        },
        session_id,
    )
    # Clean up the connection manager's in-memory dictionary
    if session_id in websocket_manager.active_connections:
        del websocket_manager.active_connections[session_id]
    if session_id in websocket_manager.active_swarms:
        del websocket_manager.active_swarms[session_id]

    return {"message": f"Session {session_id} has been closed and all data deleted."}
