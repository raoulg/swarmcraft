from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
from datetime import datetime
import logging

from swarmcraft.database.redis_client import get_redis, get_json, set_json
from swarmcraft.models.session import GameSession, SessionStatus
from swarmcraft.core.loss_functions import create_landscape
from swarmcraft.core.pso import PSO

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        # session_id -> {participant_id -> websocket}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # session_id -> PSO instance
        self.active_swarms: Dict[str, PSO] = {}

    async def connect(self, websocket: WebSocket, session_id: str, participant_id: str):
        """Connect a participant to a session"""
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}

        self.active_connections[session_id][participant_id] = websocket

        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connected",
                "message": f"Connected to session {session_id}",
                "participant_id": participant_id,
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )

        # Send current session state
        await self.send_session_state(session_id, participant_id)

        # Notify others about new participant
        await self.broadcast_to_session(
            {
                "type": "participant_joined",
                "participant_id": participant_id,
                "timestamp": datetime.now().isoformat(),
            },
            session_id,
            exclude=participant_id,
        )

        logger.info(f"Participant {participant_id} connected to session {session_id}")

    async def disconnect(self, session_id: str, participant_id: str):
        """Disconnect a participant"""
        if session_id in self.active_connections:
            if participant_id in self.active_connections[session_id]:
                del self.active_connections[session_id][participant_id]

            # Clean up empty sessions
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
                # Also clean up swarm
                if session_id in self.active_swarms:
                    del self.active_swarms[session_id]

        # Notify others about disconnection
        await self.broadcast_to_session(
            {
                "type": "participant_left",
                "participant_id": participant_id,
                "timestamp": datetime.now().isoformat(),
            },
            session_id,
        )

        logger.info(
            f"Participant {participant_id} disconnected from session {session_id}"
        )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific websocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def send_to_participant(
        self, message: dict, session_id: str, participant_id: str
    ):
        """Send message to specific participant"""
        if session_id in self.active_connections:
            if participant_id in self.active_connections[session_id]:
                websocket = self.active_connections[session_id][participant_id]
                await self.send_personal_message(message, websocket)

    async def broadcast_to_session(
        self, message: dict, session_id: str, exclude: str = None
    ):
        """Broadcast message to all participants in a session"""
        if session_id not in self.active_connections:
            return

        disconnected = []

        for participant_id, websocket in self.active_connections[session_id].items():
            if exclude and participant_id == exclude:
                continue

            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {participant_id}: {e}")
                disconnected.append(participant_id)

        # Clean up disconnected participants
        for participant_id in disconnected:
            await self.disconnect(session_id, participant_id)

    async def send_session_state(self, session_id: str, participant_id: str = None):
        """Send current session state to participant(s)"""
        redis_conn = await get_redis()
        session_data = await get_json(f"session:{session_id}", redis_conn)

        if not session_data:
            return

        session = GameSession(**session_data)
        landscape = create_landscape(
            session.config.landscape_type, **session.config.landscape_params
        )

        # Build the list of participants, now including velocity and color
        participants_list = []
        for p in session.participants:
            participants_list.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "position": p.position,
                    "fitness": p.fitness,
                    "velocity_magnitude": p.velocity_magnitude,
                    "color": landscape.get_fitness_color(
                        p.fitness, p.velocity_magnitude
                    )
                    if p.fitness is not None
                    else "#888888",
                    "connected": p.id in self.active_connections.get(session_id, {}),
                }
            )

        state_message = {
            "type": "session_state",
            "session": {
                "id": session.id,
                "code:": session.code,
                "status": session.status.value,
                "participants": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "position": p.position,
                        "fitness": p.fitness,
                        "connected": p.id
                        in self.active_connections.get(session_id, {}),
                    }
                    for p in session.participants
                ],
                "config": session.config.model_dump(mode="json"),
                "iteration": session.swarm_iteration,
            },
            "timestamp": datetime.now().isoformat(),
        }

        if participant_id:
            await self.send_to_participant(state_message, session_id, participant_id)
        else:
            await self.broadcast_to_session(state_message, session_id)

    async def handle_position_update(
        self, session_id: str, participant_id: str, position: List[int]
    ):
        """Handle participant position update with real-time feedback"""
        redis_conn = await get_redis()
        session_data = await get_json(f"session:{session_id}", redis_conn)

        if not session_data:
            return

        session = GameSession(**session_data)

        # Find and update participant
        participant = None
        for p in session.participants:
            if p.id == participant_id:
                participant = p
                break

        if not participant:
            return

        # Update position
        participant.position = position

        # Calculate fitness
        landscape = create_landscape(
            session.config.landscape_type, **session.config.landscape_params
        )

        # Convert grid to continuous coordinates
        bounds = landscape.metadata.recommended_bounds
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        grid_size = session.config.grid_size

        x = x_min + (position[1] + 0.5) * (x_max - x_min) / grid_size
        y = y_min + (position[0] + 0.5) * (y_max - y_min) / grid_size

        fitness = landscape.evaluate([x, y])
        participant.fitness = fitness

        # Save updated session
        await set_json(
            f"session:{session_id}",
            session.model_dump(mode="json"),
            redis_conn,
            expire=86400,
        )

        # Send personal feedback
        feedback_message = {
            "type": "position_feedback",
            "position": position,
            "fitness": fitness,
            "color": landscape.get_fitness_color(fitness),
            "frequency": landscape.get_fitness_audio_frequency(fitness),
            "description": landscape.describe_position([x, y]),
            "timestamp": datetime.now().isoformat(),
        }

        await self.send_to_participant(feedback_message, session_id, participant_id)

        # Broadcast position update to others
        position_update = {
            "type": "participant_moved",
            "participant_id": participant_id,
            "position": position,
            "fitness": fitness,
            "timestamp": datetime.now().isoformat(),
        }

        await self.broadcast_to_session(
            position_update, session_id, exclude=participant_id
        )

    async def run_swarm_step(self, session_id: str):
        """Execute one step of swarm optimization"""
        redis_conn = await get_redis()
        session_data = await get_json(f"session:{session_id}", redis_conn)

        if not session_data:
            return

        session = GameSession(**session_data)

        if session.status != SessionStatus.ACTIVE:
            return

        # Initialize or get existing swarm
        if session_id not in self.active_swarms:
            landscape = create_landscape(
                session.config.landscape_type, **session.config.landscape_params
            )

            # Create discrete version of landscape
            bounds = landscape.metadata.recommended_bounds
            grid_size = session.config.grid_size

            def discrete_landscape_func(position):
                # Convert continuous position to grid coordinates for evaluation
                x_min, x_max = bounds[0]
                y_min, y_max = bounds[1]

                # Clamp to bounds
                x = max(x_min, min(x_max, position[0]))
                y = max(y_min, min(y_max, position[1]))

                return landscape.evaluate([x, y])

            self.active_swarms[session_id] = PSO(
                dimensions=2,
                bounds=bounds,
                loss_function=discrete_landscape_func,
                population_size=len(session.participants),
                exploration_probability=session.config.exploration_probability,
            )

        swarm = self.active_swarms[session_id]

        # Update swarm with current participant positions
        for i, participant in enumerate(session.participants):
            if participant.position and i < len(swarm.swarm_state.particles):
                # Convert grid coordinates to continuous
                bounds = landscape.metadata.recommended_bounds
                x_min, x_max = bounds[0]
                y_min, y_max = bounds[1]
                grid_size = session.config.grid_size

                x = (
                    x_min
                    + (participant.position[1] + 0.5) * (x_max - x_min) / grid_size
                )
                y = (
                    y_min
                    + (participant.position[0] + 0.5) * (y_max - y_min) / grid_size
                )

                # Update particle position
                particle = swarm.swarm_state.particles[i]
                particle.update_position([x, y])
                if participant.fitness is not None:
                    particle.fitness = participant.fitness

        # Execute swarm step
        swarm.step()
        session.swarm_iteration += 1

        # Get swarm statistics
        stats = swarm.get_pso_statistics()

        # Save updated session
        await set_json(
            f"session:{session_id}",
            session.model_dump(mode="json"),
            redis_conn,
            expire=86400,
        )

        # Broadcast swarm update
        swarm_update = {
            "type": "swarm_update",
            "iteration": session.swarm_iteration,
            "global_best": {
                "position": stats["global_best_position"],
                "fitness": stats["global_best_fitness"],
                "particle": stats["global_best_particle"],
            },
            "statistics": {
                "diversity": stats["diversity"],
                "explorers": stats["exploration_stats"]["explorers"],
                "mean_fitness": stats["fitness_stats"]["mean"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        await self.broadcast_to_session(swarm_update, session_id)


# Global connection manager
websocket_manager = ConnectionManager()


async def handle_websocket_message(
    websocket: WebSocket, session_id: str, participant_id: str
):
    """Handle incoming WebSocket messages"""
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            message_type = message.get("type")

            if message_type == "move":
                position = message.get("position")
                if position and len(position) == 2:
                    await websocket_manager.handle_position_update(
                        session_id, participant_id, position
                    )

            elif message_type == "ping":
                await websocket_manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.now().isoformat()}, websocket
                )

            elif message_type == "get_state":
                await websocket_manager.send_session_state(session_id, participant_id)

    except WebSocketDisconnect:
        await websocket_manager.disconnect(session_id, participant_id)
    except Exception as e:
        logger.error(f"WebSocket error for {participant_id} in {session_id}: {e}")
        await websocket_manager.disconnect(session_id, participant_id)
