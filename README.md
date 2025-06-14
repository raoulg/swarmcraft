```bash
swarmcraft/
├── pyproject.toml
├── README.md
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── src/
│   └── swarmcraft/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app entry
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py           # API endpoints
│       │   └── websocket.py        # WebSocket handlers
│       ├── core/
│       │   ├── __init__.py
│       │   ├── swarm_base.py       # Base SwarmOptimizer class
│       │   ├── pso.py              # PSO implementation
│       │   └── loss_functions.py   # Rastrigin, custom functions
│       ├── models/
│       │   ├── __init__.py
│       │   ├── session.py          # Session/room models
│       │   └── participant.py     # User/participant models
│       ├── database/
│       │   ├── __init__.py
│       │   ├── sqlite_db.py        # SQLite operations
│       │   └── redis_client.py     # Redis operations
│       └── config.py               # Settings/configuration
└── frontend/                       # Future Svelte app
    └── (Phase 2)
```

curl -X POST "http://localhost:8000/api/admin/create-session" \
  -H "X-Admin-Key: $ADMIN_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "landscape_type": "rastrigin",
    "landscape_params": {"A": 10.0, "dimensions": 2},
    "grid_size": 25,
    "max_participants": 30,
    "exploration_probability": 0.15
  }'
