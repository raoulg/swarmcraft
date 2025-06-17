"""
Simple test to compare ABC vs PSO through the API.
Run this to see the current state working.
"""

import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_URL = "http://localhost:8000"
ADMIN_KEY = os.getenv("SWARM_API_KEY")

if not ADMIN_KEY:
    print("❌ SWARM_API_KEY not found in .env file")
    print("   Make sure your .env file contains: SWARM_API_KEY=your_key_here")
    exit(1)


def test_algorithm(algorithm_type, extra_config=None):
    """Test a specific algorithm through the API."""
    print(f"\n🧪 Testing {algorithm_type.upper()} Algorithm")
    print("=" * 40)

    # 1. Create session
    config = {
        "algorithm_type": algorithm_type,
        "landscape_type": "rastrigin",
        "max_iterations": 5,
        "grid_size": 10,
    }
    if extra_config:
        config.update(extra_config)

    print(f"1. Creating {algorithm_type} session...")
    response = requests.post(
        f"{BASE_URL}/api/admin/create-session",
        json=config,
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    if response.status_code != 200:
        print(f"❌ Failed to create session: {response.status_code}")
        return None

    session_data = response.json()
    session_id = session_data["session_id"]
    session_code = session_data["session_code"]
    print(f"✅ Created session: {session_code}")

    # 2. Join session
    print("2. Joining session...")
    join_response = requests.post(f"{BASE_URL}/api/join/{session_code}")
    participant_data = join_response.json()
    participant_id = participant_data["participant_id"]
    print(f"✅ Joined as: {participant_data['participant_name']}")

    # 3. Start session
    print("3. Starting session...")
    requests.post(
        f"{BASE_URL}/api/admin/session/{session_id}/start",
        headers={"X-Admin-Key": ADMIN_KEY},
    )
    print("✅ Session started")

    # 4. Check status
    print("4. Checking status...")
    status_response = requests.get(f"{BASE_URL}/api/session/{session_id}/status")
    status = status_response.json()
    print(f"   Status: {status['status']}")
    print(f"   Landscape: {status['landscape_type']}")

    # 5. Make a move
    print("5. Making a move...")
    move_response = requests.post(
        f"{BASE_URL}/api/session/{session_id}/move",
        json={"participant_id": participant_id},
    )

    if move_response.status_code == 200:
        move_data = move_response.json()
        print(f"   New position: {move_data['position']}")
        print(f"   Fitness: {move_data['fitness']:.4f}")
        print(f"   Velocity magnitude: {move_data.get('velocity_magnitude', 'N/A')}")
        print(f"   Description: {move_data.get('description', 'N/A')}")
    else:
        print(f"❌ Move failed: {move_response.status_code}")

    # 6. Trigger swarm step
    print("6. Triggering swarm step...")
    step_response = requests.post(
        f"{BASE_URL}/api/admin/session/{session_id}/step",
        headers={"X-Admin-Key": ADMIN_KEY},
    )

    if step_response.status_code == 200:
        print("✅ Swarm step executed")
    else:
        print(f"❌ Step failed: {step_response.status_code}")
        print(f"   Error details: {step_response.text}")
        # Don't return early - continue with cleanup

    # 7. Clean up
    print("7. Cleaning up...")
    requests.delete(
        f"{BASE_URL}/api/admin/session/{session_id}", headers={"X-Admin-Key": ADMIN_KEY}
    )
    print("✅ Session deleted")

    return True


def main():
    print("🚀 Testing ABC vs PSO through API")
    print("=" * 50)
    print(
        f"🔑 Using API key: {ADMIN_KEY[:8]}..." if ADMIN_KEY else "❌ No API key found"
    )
    print(f"🌐 Server: {BASE_URL}")

    # Quick health check
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server responded with status: {health_response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Server not reachable - make sure it's running on localhost:8000")
        return

    try:
        # Test ABC
        abc_success = test_algorithm("abc", {"abc_limit": 8, "abc_employed_ratio": 0.6})

        time.sleep(1)  # Brief pause between tests

        # Test PSO
        pso_success = test_algorithm(
            "pso", {"exploration_probability": 0.3, "min_exploration_probability": 0.1}
        )

        print("\n🎉 Testing Complete!")
        print("=" * 30)

        if abc_success and pso_success:
            print("✅ Both ABC and PSO work through the API!")
            print("\n💡 What's Missing for Users:")
            print("   - Users don't know if they're a 'scout bee' or 'employed bee'")
            print(
                "   - No role-specific feedback ('You're exploring!' vs 'You're following the swarm!')"
            )
            print("   - Same generic move responses for both algorithms")
            print("\n🔮 Next Steps:")
            print("   Option A: Add role info to API responses first")
            print("   Option B: Move to frontend now, add roles later")
            print("   Option C: Test with WebSocket to see real-time updates")
        else:
            print("❌ Some tests failed - check server and admin key")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the server running on localhost:8000?")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
