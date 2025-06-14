import pytest
from httpx import AsyncClient
import os

from swarmcraft.main import app
from swarmcraft.utils.name_generator import (
    generate_session_code,
    generate_participant_name,
)

# Set test environment
os.environ["SWARM_API_KEY"] = "test_admin_key_12345"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"  # Use different DB for tests


@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def admin_headers():
    return {"X-Admin-Key": "test_admin_key_12345"}


class TestAdminEndpoints:
    """Test admin-only endpoints"""

    @pytest.mark.asyncio
    async def test_create_session_success(
        self, async_client: AsyncClient, admin_headers
    ):
        """Test successful session creation"""
        session_config = {
            "landscape_type": "rastrigin",
            "landscape_params": {"A": 10.0, "dimensions": 2},
            "grid_size": 25,
            "max_participants": 20,
        }

        response = await async_client.post(
            "/api/admin/create-session", json=session_config, headers=admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "session_code" in data
        assert len(data["session_code"]) == 6
        assert "Session created!" in data["message"]

    @pytest.mark.asyncio
    async def test_create_session_no_auth(self, async_client: AsyncClient):
        """Test session creation without admin key"""
        session_config = {"landscape_type": "rastrigin"}

        response = await async_client.post(
            "/api/admin/create-session", json=session_config
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_create_session_invalid_auth(self, async_client: AsyncClient):
        """Test session creation with invalid admin key"""
        response = await async_client.post(
            "/api/admin/create-session",
            json={"landscape_type": "rastrigin"},
            headers={"X-Admin-Key": "wrong_key"},
        )

        assert response.status_code == 403


class TestParticipantEndpoints:
    """Test participant endpoints"""

    @pytest.fixture
    async def test_session(self, async_client: AsyncClient, admin_headers):
        """Create a test session"""
        session_config = {
            "landscape_type": "rastrigin",
            "landscape_params": {"A": 10.0, "dimensions": 2},
            "grid_size": 10,
            "max_participants": 5,
        }

        response = await async_client.post(
            "/api/admin/create-session", json=session_config, headers=admin_headers
        )

        return response.json()

    @pytest.mark.asyncio
    async def test_join_session_success(self, async_client: AsyncClient, test_session):
        """Test successful session join"""
        session_code = test_session["session_code"]

        response = await async_client.post(f"/api/join/{session_code}")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "participant_id" in data
        assert "participant_name" in data
        assert "Welcome" in data["message"]

    @pytest.mark.asyncio
    async def test_join_invalid_session(self, async_client: AsyncClient):
        """Test joining non-existent session"""
        response = await async_client.post("/api/join/INVALID")

        assert response.status_code == 404
        assert "Invalid session code" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_session_status(self, async_client: AsyncClient, test_session):
        """Test getting session status"""
        session_id = test_session["session_id"]

        response = await async_client.get(f"/api/session/{session_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "waiting"
        assert data["participants"] == 0
        assert data["landscape_type"] == "rastrigin"
        assert data["grid_size"] == 10

    @pytest.mark.asyncio
    async def test_make_move(self, async_client: AsyncClient, test_session):
        """Test making a move"""
        session_code = test_session["session_code"]
        session_id = test_session["session_id"]

        # First join the session
        join_response = await async_client.post(f"/api/join/{session_code}")
        participant_id = join_response.json()["participant_id"]

        # Make a move
        move_data = {
            "participant_id": participant_id,
            "position": [5, 5],  # Center of 10x10 grid
        }

        response = await async_client.post(
            f"/api/session/{session_id}/move", json=move_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["position"] == [5, 5]
        assert "fitness" in data
        assert "color" in data
        assert "frequency" in data
        assert "description" in data

    @pytest.mark.asyncio
    async def test_move_out_of_bounds(self, async_client: AsyncClient, test_session):
        """Test invalid move"""
        session_code = test_session["session_code"]
        session_id = test_session["session_id"]

        # Join session
        join_response = await async_client.post(f"/api/join/{session_code}")
        participant_id = join_response.json()["participant_id"]

        # Try invalid move
        move_data = {
            "participant_id": participant_id,
            "position": [15, 15],  # Out of bounds for 10x10 grid
        }

        response = await async_client.post(
            f"/api/session/{session_id}/move", json=move_data
        )

        assert response.status_code == 400
        assert "out of bounds" in response.json()["detail"]


class TestUtilities:
    """Test utility functions"""

    def test_generate_session_code(self):
        """Test session code generation"""
        code = generate_session_code()
        assert len(code) == 6
        assert code.isalnum()
        assert "O" not in code  # Should avoid confusing characters
        assert "0" not in code
        assert "I" not in code
        assert "1" not in code

    def test_generate_participant_name(self):
        """Test participant name generation"""
        name = generate_participant_name()
        assert " " in name  # Should have space between adjective and animal
        parts = name.split(" ")
        assert len(parts) == 2
        assert len(parts[0]) > 2  # Reasonable adjective length
        assert len(parts[1]) > 2  # Reasonable animal length

    def test_unique_names(self):
        """Test that names are reasonably unique"""
        names = [generate_participant_name() for _ in range(100)]
        unique_names = set(names)
        # Should have good variety (at least 80% unique in 100 generations)
        assert len(unique_names) > 80


class TestLandscapeIntegration:
    """Test landscape integration in API"""

    @pytest.mark.asyncio
    async def test_rastrigin_feedback(self, async_client: AsyncClient, admin_headers):
        """Test Rastrigin landscape feedback"""
        # Create Rastrigin session
        session_config = {
            "landscape_type": "rastrigin",
            "landscape_params": {"A": 10.0, "dimensions": 2},
            "grid_size": 25,
        }

        session_response = await async_client.post(
            "/api/admin/create-session", json=session_config, headers=admin_headers
        )

        session_data = session_response.json()
        session_code = session_data["session_code"]
        session_id = session_data["session_id"]

        # Join session
        join_response = await async_client.post(f"/api/join/{session_code}")
        participant_id = join_response.json()["participant_id"]

        # Test center position (should be near global minimum)
        center_move = {
            "participant_id": participant_id,
            "position": [12, 12],  # Center of 25x25 grid
        }

        center_response = await async_client.post(
            f"/api/session/{session_id}/move", json=center_move
        )

        center_data = center_response.json()
        center_fitness = center_data["fitness"]

        # Test corner position (should be worse)
        corner_move = {
            "participant_id": participant_id,
            "position": [0, 0],  # Corner
        }

        corner_response = await async_client.post(
            f"/api/session/{session_id}/move", json=corner_move
        )

        corner_data = corner_response.json()
        corner_fitness = corner_data["fitness"]

        # Center should be better (lower fitness) than corner
        assert center_fitness < corner_fitness

        # Check feedback format
        assert center_data["color"].startswith("#")
        assert len(center_data["color"]) == 7
        assert 200 <= center_data["frequency"] <= 800
        assert isinstance(center_data["description"], str)
        assert len(center_data["description"]) > 10


# WebSocket tests (more complex, using pytest-asyncio)
class TestWebSocket:
    """Test WebSocket functionality"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        # Note: This would need a more complex setup with actual WebSocket testing
        # For now, just test that the endpoint exists
        pass  # TODO: Implement WebSocket testing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
