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
    JoinRequest,
)
from swarmcraft.utils.name_generator import (
    generate_participant_name,
    generate_session_code,
)
from swarmcraft.core.loss_functions import create_landscape
from swarmcraft.core.swarm_base import SwarmState, Particle
from swarmcraft.core.pso import PSO
from swarmcraft.api.websocket import websocket_manager
from swarmcraft.core.algorithm_factory import create_optimizer
from swarmcraft.models.session import AlgorithmType
from swarmcraft.core.algorithm_factory import (
    get_algorithm_display_name,
    get_algorithm_description,
)
from loguru import logger

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
async def join_session(
    session_code: str,
    join_request: JoinRequest = JoinRequest(),
    redis_conn=Depends(get_redis),
):
    """Join a session using session code. Supports reconnection with participant_id."""
    # Get session ID from code
    logger.info(
        f"Join request for session code {session_code}, participant_id: {join_request.participant_id}"
    )
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

    # Check if reconnecting to existing participant
    participant = None
    is_reconnect = False

    if join_request.participant_id:
        # Try to find existing participant
        for p in session.participants:
            if p.id == join_request.participant_id:
                participant = p
                participant.connected = True
                is_reconnect = True
                logger.info(
                    f"Reconnecting participant {participant.id} ({participant.name})"
                )
                break

        # Log if reconnection failed
        if not participant:
            logger.warning(
                f"Reconnection failed: participant_id {join_request.participant_id} not found in session {session_id}"
            )

    # If not reconnecting, create new participant
    if not participant:
        if len(session.participants) >= session.config.max_participants:
            raise HTTPException(status_code=400, detail="Session full")

        # Create new participant
        participant_id = f"p_{len(session.participants) + 1}"
        name, emojis = generate_participant_name()
        participant = Participant(
            id=participant_id,
            name=name,
            emojis=emojis,
            joined_at=datetime.now(),
        )

        # Add to session
        session.participants.append(participant)
        logger.info(f"Created new participant {participant.id} ({participant.name})")

    # ===== Handle active sessions (only for NEW participants, not reconnects) =====
    if session.status == SessionStatus.ACTIVE and not is_reconnect:
        # Load existing swarm state
        swarm_state_data = await get_json(f"swarm_state:{session_id}", redis_conn)
        if swarm_state_data:
            # Create PSO instance with updated participant count
            landscape = create_landscape(
                session.config.landscape_type,
                grid_size=session.config.grid_size,
                **session.config.landscape_params,
            )
            pso = PSO(
                dimensions=2,
                bounds=landscape.metadata.recommended_bounds,
                loss_function=landscape.evaluate,
                population_size=len(session.participants),  # Updated count
                max_iterations=session.config.max_iterations,
                exploration_probability=session.config.exploration_probability,
                min_exploration_probability=session.config.min_exploration_probability,
            )

            # Restore existing swarm state
            pso.swarm_state = SwarmState(**swarm_state_data)

            # Add a new particle for the new participant
            new_particle_id = f"particle_{len(pso.swarm_state.particles)}"
            new_position = pso._generate_random_position()
            new_fitness = landscape.evaluate(new_position)

            new_particle = Particle(
                id=new_particle_id,
                position=new_position.tolist(),
                velocity=[0.0] * pso.dimensions,
                fitness=new_fitness,
                personal_best_position=new_position.tolist(),
                personal_best_fitness=new_fitness,
            )

            # Add to swarm state
            pso.swarm_state.particles.append(new_particle)

            # Update global best if this new particle is better
            if new_fitness < pso.swarm_state.global_best_fitness:
                pso.swarm_state.update_global_best(
                    new_position, new_fitness, new_particle_id
                )

            # Sync all participants with swarm (includes new participant)
            session.participants = pso.sync_participants_from_swarm(
                participants=session.participants, grid_size=session.config.grid_size
            )

            # Save updated swarm state
            await set_json(
                f"swarm_state:{session_id}",
                pso.swarm_state.model_dump(mode="json"),
                redis_conn,
                expire=86400,
            )

    # Save updated session
    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )

    # ===== NEW: Broadcast participant joined =====
    # Build participants with colors for broadcasting
    participants_with_color = []
    if session.status == SessionStatus.ACTIVE:
        landscape = create_landscape(
            session.config.landscape_type,
            grid_size=session.config.grid_size,
            **session.config.landscape_params,
        )
        for p in session.participants:
            p_dict = p.model_dump(mode="json")
            p_dict["color"] = (
                landscape.get_fitness_color(p.fitness, p.velocity_magnitude)
                if p.fitness is not None
                else "#888888"
            )
            participants_with_color.append(p_dict)
    else:
        participants_with_color = [
            p.model_dump(mode="json") for p in session.participants
        ]

    # Broadcast participant joined/reconnected
    message_type = "participant_reconnected" if is_reconnect else "participant_joined"
    await websocket_manager.broadcast_to_session(
        {
            "type": message_type,
            "participant_id": participant.id,
            "participant_name": participant.name,
            "participants": participants_with_color,  # Include full updated list
            "session": session.model_dump(mode="json"),
            "timestamp": datetime.now().isoformat(),
        },
        session_id,
    )
    logger.info(f"Session status {session.status}")
    logger.info(f"Participant count: {len(session.participants)}")

    message = (
        f"Welcome back, {participant.name}!"
        if is_reconnect
        else f"Welcome, {participant.name}!"
    )
    return {
        "session_id": session_id,
        "participant_id": participant.id,
        "participant_name": participant.name,
        "is_reconnect": is_reconnect,
        "message": message,
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

    # 2. HYDRATE a temporary optimizer "calculator" with the loaded state.
    landscape = create_landscape(
        session.config.landscape_type,
        grid_size=session.config.grid_size,
        **session.config.landscape_params,
    )

    # CHANGED: Use factory instead of hardcoded PSO
    optimizer = create_optimizer(
        config=session.config,
        participants_count=len(session.participants),
        landscape=landscape,
    )

    # This is the crucial step: restoring the full state of the swarm.
    optimizer.swarm_state = swarm_state

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
    optimizer_particle_id = f"particle_{participant_index}"

    # 4. MUTATE the state by calculating the particle's next move.
    if hasattr(optimizer, "suggest_next_position"):
        suggested_cont_pos = optimizer.suggest_next_position(optimizer_particle_id)
    else:
        # Fallback for optimizers that don't have suggest_next_position
        raise HTTPException(
            status_code=500, detail="Optimizer does not support position suggestions."
        )

    if suggested_cont_pos is None:
        raise HTTPException(
            status_code=500, detail="Could not calculate next position."
        )

    # 5. Update the state of the specific particle within the swarm_state object.
    particle_to_update = optimizer.get_particle_by_id(optimizer_particle_id)
    new_fitness = landscape.evaluate(np.array(suggested_cont_pos))
    velocity_mag = 0.0
    if particle_to_update:
        particle_to_update.update_position(np.array(suggested_cont_pos))
        particle_to_update.fitness = new_fitness
        velocity_mag = float(np.linalg.norm(particle_to_update.velocity))

    # 6. Convert the new continuous position back to a discrete grid position.
    bounds = landscape.metadata.recommended_bounds
    grid_size = session.config.grid_size
    new_x, new_y = suggested_cont_pos[0], suggested_cont_pos[1]
    col = int((new_x - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) * grid_size)
    row = int((new_y - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) * grid_size)
    new_grid_pos = [max(0, min(grid_size - 1, col)), max(0, min(grid_size - 1, row))]

    # 7. Update the participant's info in the GameSession object.
    session.participants[participant_index].position = new_grid_pos
    session.participants[participant_index].fitness = new_fitness
    session.participants[participant_index].velocity_magnitude = velocity_mag

    # 8. SAVE both updated states back to Redis.
    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),  # CHANGED: use mode="json"
        redis_conn,
        expire=86400,
    )
    await set_json(
        f"swarm_state:{session_id}",
        optimizer.swarm_state.model_dump(mode="json"),  # CHANGED: use mode="json"
        redis_conn,
        expire=86400,
    )

    # 9. Return feedback to the user.
    return {
        "position": new_grid_pos,
        "fitness": new_fitness,
        "velocity_magnitude": velocity_mag,
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
    particles at once and broadcasting the result.
    """
    session_data = await get_json(f"session:{session_id}", redis_conn)
    swarm_state_data = await get_json(f"swarm_state:{session_id}", redis_conn)

    if not session_data or not swarm_state_data:
        raise HTTPException(status_code=404, detail="Session or Swarm state not found.")

    session = GameSession(**session_data)
    swarm_state = SwarmState(**swarm_state_data)

    if session.status != SessionStatus.ACTIVE:
        return {"message": "Swarm step skipped: session is not active."}

    landscape = create_landscape(
        session.config.landscape_type,
        grid_size=session.config.grid_size,
        **session.config.landscape_params,
    )

    # CHANGED: Use factory instead of hardcoded PSO
    optimizer = create_optimizer(
        config=session.config,
        participants_count=len(session.participants),
        landscape=landscape,
    )
    optimizer.swarm_state = swarm_state

    optimizer.step()

    updated_participants = optimizer.sync_participants_from_swarm(
        participants=session.participants, grid_size=session.config.grid_size
    )
    session.participants = updated_participants
    session.swarm_iteration = optimizer.swarm_state.iteration

    # Get algorithm-appropriate statistics
    if hasattr(optimizer, "get_pso_statistics"):
        stats = optimizer.get_pso_statistics()  # PSO has detailed stats
    elif hasattr(optimizer, "get_abc_statistics"):
        stats = optimizer.get_abc_statistics()  # ABC has detailed stats
    else:
        stats = optimizer.get_swarm_statistics()  # Fallback to base stats

    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),  # CHANGED: use mode="json"
        redis_conn,
        expire=86400,
    )
    await set_json(
        f"swarm_state:{session_id}",
        optimizer.swarm_state.model_dump(mode="json"),  # CHANGED: use mode="json"
        redis_conn,
        expire=86400,
    )

    # Build the participant list with colors
    participants_with_color = []
    for p in updated_participants:
        p_dict = p.model_dump(mode="json")  # CHANGED: use mode="json"
        p_dict["color"] = (
            landscape.get_fitness_color(p.fitness, p.velocity_magnitude)
            if p.fitness is not None
            else "#888888"
        )
        participants_with_color.append(p_dict)

    swarm_update_message = {
        "type": "swarm_update",
        "iteration": session.swarm_iteration,
        "participants": participants_with_color,
        "statistics": {
            "global_best_fitness": stats["global_best_fitness"],
            # CHANGED: Handle different statistics based on algorithm
            "explorers": stats.get("exploration_stats", {}).get("explorers", 0),
            "exploration_probability": stats.get(
                "current_exploration_probability", 0.0
            ),
        },
        "timestamp": datetime.now().isoformat(),
    }
    await websocket_manager.broadcast_to_session(swarm_update_message, session_id)

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

    # 2. CREATE the optimizer using the factory (CHANGED: was hardcoded PSO)
    landscape = create_landscape(
        session.config.landscape_type,
        grid_size=session.config.grid_size,
        **session.config.landscape_params,
    )

    optimizer = create_optimizer(
        config=session.config,
        participants_count=len(session.participants),
        landscape=landscape,
    )

    # 3. SYNC the game state with the newly created swarm state.
    session.participants = optimizer.sync_participants_from_swarm(
        participants=session.participants, grid_size=session.config.grid_size
    )

    # 4. SAVE the updated states back to Redis
    await set_json(
        f"session:{session_id}",
        session.model_dump(
            mode="json"
        ),  # CHANGED: use mode="json" for enum serialization
        redis_conn,
        expire=86400,
    )
    await set_json(
        f"swarm_state:{session_id}",
        optimizer.swarm_state.model_dump(mode="json"),  # CHANGED: use mode="json"
        redis_conn,
        expire=86400,
    )

    # 5. NOTIFY clients that the game is on!
    await websocket_manager.broadcast_to_session(
        {
            "type": "session_started",
            "message": "The swarm optimization has begun! You have been assigned a starting position.",
            "timestamp": datetime.now().isoformat(),
            "session": session.model_dump(mode="json"),  # CHANGED: use mode="json"
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


@router.post("/admin/session/{session_id}/reset")
async def reset_session(
    session_id: str, redis_conn=Depends(get_redis), _: bool = Depends(verify_admin_key)
):
    """
    Resets a session to its pre-start state while keeping all participants.
    It resets the iteration count, clears swarm state, and sets the status to WAITING.
    """
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = GameSession(**session_data)

    # Reset core attributes
    session.status = SessionStatus.WAITING
    session.swarm_iteration = 0
    session.started_at = None

    # Clear position and fitness data for each participant
    for p in session.participants:
        p.position = None
        p.fitness = None
        p.velocity_magnitude = None

    # Save the reset session state
    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )

    # Delete the persistent swarm state from Redis
    await redis_conn.delete(f"swarm_state:{session_id}")

    # Notify all clients that the session has been reset
    await websocket_manager.broadcast_to_session(
        {
            "type": "session_reset",
            "message": "The session has been reset by the admin.",
            "session": session.model_dump(mode="json"),
        },
        session_id,
    )

    return {"message": f"Session {session_id} has been reset."}


@router.post("/admin/session/{session_id}/remove-participant")
async def remove_participant(
    session_id: str,
    participant_data: dict,  # Should contain participant_id
    redis_conn=Depends(get_redis),
    _: bool = Depends(verify_admin_key),
):
    """Remove a participant from the session (admin only)"""
    participant_id = participant_data.get("participant_id")
    if not participant_id:
        raise HTTPException(status_code=400, detail="participant_id required")

    # Get session data
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)

    # Remove participant from the list
    session.participants = [p for p in session.participants if p.id != participant_id]

    # Save updated session
    await set_json(
        f"session:{session_id}",
        session.model_dump(mode="json"),
        redis_conn,
        expire=86400,
    )

    # Notify remaining participants
    await websocket_manager.broadcast_to_session(
        {
            "type": "participant_removed",
            "participant_id": participant_id,
            "message": f"Participant {participant_id} has been removed",
            "participants": [p.model_dump(mode="json") for p in session.participants],
        },
        session_id,
    )

    return {"message": f"Participant {participant_id} removed successfully"}


@router.post("/admin/session/{session_id}/reveal")
async def trigger_reveal(
    session_id: str, redis_conn=Depends(get_redis), _: bool = Depends(verify_admin_key)
):
    """
    Triggers a reveal phase, showing participants their fitness scores
    """
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)

    if session.status != SessionStatus.ACTIVE:
        return {"message": "Reveal skipped: session is not active."}

    # Broadcast reveal message to all participants
    reveal_message = {
        "type": "reveal_fitness",
        "message": "Revealing your fitness scores!",
        "timestamp": datetime.now().isoformat(),
    }
    await websocket_manager.broadcast_to_session(reveal_message, session_id)

    return {"message": "Fitness reveal triggered successfully."}


@router.get("/session/{session_id}/landscape")
async def get_session_landscape(
    session_id: str, resolution: int = 100, redis_conn=Depends(get_redis)
):
    """
    Get the landscape values for visualization.
    Returns a flattened array of values matching the frontend grid (row-major).
    """
    session_data = await get_json(f"session:{session_id}", redis_conn)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = GameSession(**session_data)

    landscape = create_landscape(
        session.config.landscape_type,
        grid_size=session.config.grid_size,
        **session.config.landscape_params,
    )

    bounds = landscape.metadata.recommended_bounds

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Use the requested resolution for visualization, not the game grid size
    viz_grid_size = resolution

    x_step = (x_max - x_min) / viz_grid_size
    y_step = (y_max - y_min) / viz_grid_size

    values = []

    # Calculate values for each cell to match frontend grid order
    # Frontend iterates i from 0 to grid_size*grid_size - 1
    # row = floor(i / grid_size) (0 is top)
    # col = i % grid_size (0 is left)
    for i in range(viz_grid_size * viz_grid_size):
        visual_row = i // viz_grid_size
        visual_col = i % viz_grid_size

        # Convert to logical grid coordinates and then continuous space
        # Visual row 0 (top) -> Logical Y (grid_size - 1)
        logical_y_idx = viz_grid_size - 1 - visual_row
        logical_x_idx = visual_col

        # Center of the cell
        x = x_min + (logical_x_idx + 0.5) * x_step
        y = y_min + (logical_y_idx + 0.5) * y_step

        val = landscape.evaluate(np.array([x, y]))
        values.append(float(val))

    return {
        "grid_size": viz_grid_size,
        "values": values,
        "min_value": min(values),
        "max_value": max(values),
    }


# Add this new route to your src/swarmcraft/api/routes.py file


@router.get("/algorithms")
async def get_available_algorithms():
    """Get list of available swarm algorithms with their parameter schemas"""

    algorithms = []
    for algo_type in AlgorithmType:
        algorithms.append(
            {
                "id": algo_type.value,
                "name": get_algorithm_display_name(algo_type),
                "description": get_algorithm_description(algo_type),
                "parameter_schema": _get_algorithm_parameter_schema(algo_type),
            }
        )

    return {"algorithms": algorithms}


def _get_algorithm_parameter_schema(algorithm_type: AlgorithmType) -> dict:
    """Get the parameter schema (not values) for each algorithm type"""
    if algorithm_type == AlgorithmType.PSO:
        return {
            "exploration_probability": {
                "type": "number",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Initial probability of random exploration",
            },
            "min_exploration_probability": {
                "type": "number",
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "description": "Minimum exploration probability (annealing target)",
            },
        }
    elif algorithm_type == AlgorithmType.ABC:
        return {
            "abc_limit": {
                "type": "integer",
                "default": 10,
                "min": 1,
                "max": 50,
                "step": 1,
                "description": "Abandonment limit for food sources",
            },
            "abc_employed_ratio": {
                "type": "number",
                "default": 0.5,
                "min": 0.1,
                "max": 0.9,
                "step": 0.1,
                "description": "Ratio of employed bees to total population",
            },
        }
    else:
        return {}
