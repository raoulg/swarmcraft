# SwarmCraft
*Interactive Swarm Intelligence for Experiential Learning*

## Project Overview

SwarmCraft is an innovative educational platform that makes swarm intelligence viscerally understandable through embodied, interactive experiences. Participants become "particles" in optimization algorithms, experiencing firsthand how collective intelligence emerges from individual actions and discovering the crucial balance between exploration and exploitation.

### Core Philosophy

- **Experiential Learning**: Mathematical concepts become visceral through physical participation.
- **Multilayered Teaching**: Multiple levels of meaning accessible to different audiences (13-year-olds to policymakers).
- **Human-AI Collaboration**: Demonstrates emergent intelligence through human-AI partnership.
- **Embodied Understanding**: Participants physically feel algorithmic forces and optimization landscapes.

### Target Audiences

- **High School Students (13+)**: Understanding collective behavior and emergence.
- **Graduate Students**: Applied data science and optimization concepts.
- **Policymakers**: Coordination problems, local vs. global optima, and systemic thinking.
- **Researchers**: Human-AI collaboration and swarm intelligence applications.

## System Architecture

### Technology Stack

**Backend:**
- **FastAPI**: Modern async web framework with automatic API documentation.
- **Redis**: Real-time session state management and participant coordination.
- **Pydantic V2**: Robust data validation and serialization.
- **WebSockets**: Real-time bidirectional communication for live updates.
- **Loguru**: Clear, timestamped, and colorized logging for easy debugging.

**Core Algorithms:**
- **PSO (Particle Swarm Optimization)**: Primary swarm intelligence algorithm.
- **Adaptive Parameters**: Dynamic exploration/exploitation balance with annealing.
- **Grid Discretization**: Continuous optimization mapped to discrete human interaction.

**Visualization & Testing:**
- **Plotly**: Interactive 3D/2D landscape visualization for analysis.
- **Jupyter Notebooks**: Rapid prototyping and algorithm tuning suite.
- **HTML/JS Test Client**: A minimal client for end-to-end testing of the full application loop.

## Implementation Roadmap

### ✅ **Phase 1: Core Algorithm Framework** (COMPLETED)

The underlying swarm intelligence engine and visualization tools are complete and tested.

### ✅ **Phase 2: Backend API Infrastructure** (COMPLETED)

The FastAPI application is fully functional with robust, stateful session management using Redis and admin-protected endpoints.

### ✅ **Phase 3: End-to-End Testing & Validation** (COMPLETED)

The system has been validated from end to end using a minimal HTML/JS test client to verify the full application loop.

### ✅ **Phase 4: Algorithm Tuning & Refinement** (COMPLETED)

The PSO algorithm has been refined with a new test landscape (`Quadratic`), exploration annealing for smoother convergence, and enhanced debugging via improved logging and a more flexible test client. All core logic has been unit tested.

### 🚧 **Phase 5: Mobile Frontend Development (Svelte)** (IN PROGRESS)

With a stable and tested backend, the next major step is to build the mobile-first frontend experience for participants.
- [ ] **Setup SvelteKit Project**: Initialize the frontend project with TypeScript and TailwindCSS.
- [ ] **Session Joining Flow**: Create UI for joining a session via a code.
- [ ] **Grid Interface**: Develop the primary 25x25 grid with touch interaction.
- [ ] **WebSocket Integration**: Connect the frontend to the backend for live updates.
- [ ] **Real-time Visuals**: Implement real-time updates for particle positions and card colors based on WebSocket messages.
- [ ] **Participant View**: Create the individual participant view that will be the core of the mobile experience.
- [ ] **Testing Strategy**:
    1. **Multiple Browser Windows**: Use incognito tabs to simulate different users joining the same session.
    2. **Single-Screen Test Harness (Recommended)**: Build a special Svelte page that can host multiple participant components at once. This makes it easy to visualize how broadcasted events affect all users simultaneously.
    3. **Automated E2E Testing (Future)**: Plan for using a framework like Playwright or Cypress to script and automate multi-user test scenarios.

### 🔬 **Phase 6: Advanced Algorithms & Social Dynamics** (PLANNED)

This phase focuses on enriching the experience by introducing more complex and socially interesting algorithms.
- [ ] **Implement New Algorithms**:
  - [ ] **Artificial Bee Colony (ABC)**: Introduce roles like "employed," "onlooker," and "scout" bees.
  - [ ] **Grey Wolf Optimizer (GWO)**: Implement a social hierarchy with "alpha," "beta," and "omega" wolves.
- [ ] **Broadcast Participant Roles**: Update the WebSocket communication protocol to broadcast the specific role of each participant (e.g., `role: "scout"` or `role: "alpha"`) so the frontend can display it.
- [ ] **Role-Based UI**: Modify the frontend to visually represent these different roles, enhancing the social learning aspect.

### 📋 **Phase 7: Enhanced User Experience** (PLANNED)

**Advanced Feedback Systems:**
- [ ] Advanced audio synthesis with harmonic progression based on swarm fitness.
- [ ] Haptic feedback integration for mobile devices.
- [ ] Visual effects and particle animations.
- [ ] Collaborative achievement system and progress indicators.

**Educational Content & Admin Tools:**
- [ ] Interactive tutorials explaining swarm intelligence principles.
- [ ] Post-session analysis and reflection tools for participants.
- [ ] An admin dashboard for real-time session monitoring and control.

### 🤖 **Phase 8: Human-AI Collaboration Suite** (VISION)

**Existential Meaning Navigation:**
- [ ] Personal growth optimization with AI guidance.
- [ ] Meaning-making through collaborative intelligence.
- [ ] Therapeutic applications with professional oversight.

## Getting Started

### Prerequisites

- Python 3.12
- uv package manager
- Redis (Docker or local installation)
- OpenSSL (for admin key generation)

### Quick Setup

```bash
# Clone and setup project
git clone <repository-url>
cd swarmcraft
uv sync

# Generate admin key and setup environment
echo "SWARM_API_KEY=$(openssl rand -hex 32)" > .env
echo "REDIS_URL=redis://localhost:6379" >> .env

# Start services
docker compose up --build

# Test the system
source .env
curl http://localhost:8000/health
```

dev workflow
```bash
# Start services
docker compose up

# Make changes to Python files. The server will auto-reload.
# No need to rebuild unless changing dependencies (pyproject.toml).

# Run tests
uv run pytest tests/ -v

# API documentation
open http://localhost:8000/docs
```
