import asyncio
import websockets
import json

# --- CONFIGURE THESE ---
SESSION_ID = "LC7K83"
PARTICIPANT_ID = "p_1"  # Use the ID you get after joining
# ---------------------


async def listen_to_swarm():
    uri = f"ws://localhost:8000/ws/{SESSION_ID}/{PARTICIPANT_ID}"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to session {SESSION_ID} as {PARTICIPANT_ID}")
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                print("\n--- MESSAGE RECEIVED ---")
                print(json.dumps(data, indent=2))
                print("------------------------\n")
            except websockets.ConnectionClosed:
                print("Connection closed.")
                break


if __name__ == "__main__":
    asyncio.run(listen_to_swarm())
