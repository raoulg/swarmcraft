from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
from swarmcraft.config import (
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_PARTICIPANTS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_EXPLORATION_PROBABILITY,
    DEFAULT_LANDSCAPE_TYPE,
)


class SessionStatus(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class AlgorithmType(Enum):
    """Supported swarm optimization algorithms."""

    PSO = "pso"
    ABC = "abc"


class Participant(BaseModel):
    id: str
    name: str
    emojis: Optional[List[str]] = []
    position: Optional[List[int]] = None  # Grid coordinates [i, j]
    continuous_position: Optional[List[float]] = (
        None  # Actual continuous coordinates [x, y]
    )
    normalized_position: Optional[List[float]] = (
        None  # Normalized continuous coordinates [0..1, 0..1]
    )
    fitness: Optional[float] = None
    velocity_magnitude: Optional[float] = None
    connected: bool = True
    joined_at: datetime


class SessionConfig(BaseModel):
    # Existing fields (unchanged)
    landscape_type: str = DEFAULT_LANDSCAPE_TYPE
    landscape_params: Dict = {}
    grid_size: int = DEFAULT_GRID_SIZE
    max_participants: int = DEFAULT_MAX_PARTICIPANTS
    exploration_probability: float = DEFAULT_EXPLORATION_PROBABILITY
    min_exploration_probability: Optional[float] = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    # NEW: Algorithm selection (using enum for safety)
    algorithm_type: AlgorithmType = AlgorithmType.PSO

    # NEW: ABC-specific parameters (optional, with sensible defaults)
    abc_limit: Optional[int] = 10  # Abandonment limit for ABC
    abc_employed_ratio: Optional[float] = 0.5  # Fraction of employed bees


class GameSession(BaseModel):
    id: str
    code: str
    admin_id: str
    status: SessionStatus = SessionStatus.WAITING
    config: SessionConfig
    participants: List[Participant] = []
    swarm_iteration: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None


class MoveData(BaseModel):
    participant_id: str


class JoinRequest(BaseModel):
    """Request body for joining a session"""

    participant_id: Optional[str] = (
        None  # If provided, reconnect to existing participant
    )
