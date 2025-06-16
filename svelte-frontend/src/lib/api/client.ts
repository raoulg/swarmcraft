import { sessionState, participantId, logEvent } from '$lib/stores/sessionStore';

const API_BASE_URL = 'http://localhost:8000/api';
const WS_BASE_URL = 'ws://localhost:8000';

let websocket: WebSocket | null = null;

// --- WebSocket Management ---

export function connectWebSocket(sessionId: string, partId: string) {
	if (websocket) {
		websocket.close();
	}

	const url = `${WS_BASE_URL}/ws/${sessionId}/${partId}`;
	websocket = new WebSocket(url);

	websocket.onopen = () => {
		logEvent(`WebSocket connected for ${partId}`);
	};

	websocket.onmessage = (event) => {
		const data = JSON.parse(event.data);
		logEvent(`WS Received: ${data.type}`, data);

		// Handle different message types from the backend
		switch (data.type) {
			case 'session_state':
			case 'session_started':
			case 'session_reset':
				sessionState.set(data.session);
				break;
			case 'swarm_update':
				sessionState.update((s) => ({
					...s,
					iteration: data.iteration,
					participants: data.participants
				}));
				break;
			case 'participant_moved':
				sessionState.update((s) => ({
					...s,
					participants: s.participants.map((p) =>
						p.id === data.participant_id
							? { ...p, position: data.position, fitness: data.fitness }
							: p
					)
				}));
				break;
			case 'session_closed':
				alert('The session has been closed by the admin.');
				websocket?.close();
				sessionState.set({ participants: [] });
				break;
		}
	};

	websocket.onclose = () => {
		logEvent('WebSocket disconnected');
		websocket = null;
	};

	websocket.onerror = (err) => {
		logEvent('WebSocket error', err);
	};
}

// --- API Functions ---

export async function createSession(adminKey: string, landscape: string, iterations: number) {
	const response = await fetch(`${API_BASE_URL}/admin/create-session`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json', 'X-Admin-Key': adminKey },
		body: JSON.stringify({
			landscape_type: landscape,
			grid_size: 25,
			max_participants: 50,
			max_iterations: iterations,
			min_exploration_probability: 0.01,
			exploration_probability: 0.2
		})
	});
	if (!response.ok) throw new Error('Failed to create session');
	return response.json();
}

export async function joinSession(sessionCode: string) {
	const response = await fetch(`${API_BASE_URL}/join/${sessionCode}`, { method: 'POST' });
	if (!response.ok) throw new Error('Failed to join session');
	const data = await response.json();
	participantId.set(data.participant_id);
	// Connect to WebSocket after successfully joining
	connectWebSocket(data.session_id, data.participant_id);
	return data;
}

export async function startSession(adminKey: string, sessionId: string) {
	const response = await fetch(`${API_BASE_URL}/admin/session/${sessionId}/start`, {
		method: 'POST',
		headers: { 'X-Admin-Key': adminKey }
	});
	if (!response.ok) throw new Error('Failed to start session');
	return response.json();
}

export async function triggerStep(adminKey: string, sessionId: string) {
	const response = await fetch(`${API_BASE_URL}/admin/session/${sessionId}/step`, {
		method: 'POST',
		headers: { 'X-Admin-Key': adminKey }
	});
	if (!response.ok) throw new Error('Failed to trigger step');
	return response.json();
}

export async function resetSession(adminKey: string, sessionId: string) {
	const response = await fetch(`${API_BASE_URL}/admin/session/${sessionId}/reset`, {
		method: 'POST',
		headers: { 'X-Admin-Key': adminKey }
	});
	if (!response.ok) throw new Error('Failed to reset session');
	return response.json();
}
