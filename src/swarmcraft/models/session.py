from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


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
    position: Optional[List[int]] = None  # Grid coordinates [i, j]
    fitness: Optional[float] = None
    velocity_magnitude: Optional[float] = None
    connected: bool = True
    joined_at: datetime


class SessionConfig(BaseModel):
    # Existing fields (unchanged)
    landscape_type: str = "rastrigin"
    landscape_params: Dict = {}
    grid_size: int = 25
    max_participants: int = 30
    exploration_probability: float = 0.15
    min_exploration_probability: Optional[float] = None
    max_iterations: int = 10

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
