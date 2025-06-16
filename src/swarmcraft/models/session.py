from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class SessionStatus(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class Participant(BaseModel):
    id: str
    name: str
    position: Optional[List[int]] = None  # Grid coordinates [i, j]
    fitness: Optional[float] = None
    velocity_magnitude: Optional[float] = None
    connected: bool = True
    joined_at: datetime


class SessionConfig(BaseModel):
    landscape_type: str = "rastrigin"
    landscape_params: Dict = {}
    grid_size: int = 25
    max_participants: int = 30
    exploration_probability: float = 0.15
    min_exploration_probability: Optional[float] = None
    max_iterations: int = 10


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
