<script lang="ts">
	import { createSession, startSession, triggerStep, resetSession } from '$lib/api/client';
	import { sessionState } from '$lib/stores/sessionStore';

	let adminKey = '';
	let landscapeType = 'quadratic';
	let maxIterations = 20;
	let isLoading = false;

	async function handleCreateSession() {
		isLoading = true;
		try {
			const data = await createSession(adminKey, landscapeType, maxIterations);
			alert(`Session created! Code: ${data.session_code}, ID: ${data.session_id}`);
			// Admin joins their own session to get updates
			await joinSessionAsAdmin(data.session_id);
		} catch (e) {
			alert(e.message);
		}
		isLoading = false;
	}

	// Admin needs to "join" to be part of the WebSocket broadcast group
	async function joinSessionAsAdmin(sessionId: string) {
		const partId = `admin_${Math.random().toString(36).substring(2, 8)}`;
		connectWebSocket(sessionId, partId);
		const statusRes = await fetch(`http://localhost:8000/api/session/${sessionId}/status`);
		const data = await statusRes.json();
		sessionState.update((s) => ({
			...s,
			id: sessionId,
			config: { grid_size: data.grid_size, landscape_type: data.landscape_type }
		}));
	}

	async function handleAction(action: 'start' | 'step' | 'reset') {
		if (!$sessionState.id) {
			alert('No active session ID.');
			return;
		}
		isLoading = true;
		try {
			switch (action) {
				case 'start':
					await startSession(adminKey, $sessionState.id);
					break;
				case 'step':
					await triggerStep(adminKey, $sessionState.id);
					break;
				case 'reset':
					await resetSession(adminKey, $sessionState.id);
					break;
			}
		} catch (e) {
			alert(e.message);
		}
		isLoading = false;
	}
</script>

<div class="card w-full flex flex-col gap-4">
	<h2 class="text-2xl font-bold text-accent-color">Admin Controls</h2>

	<div class="flex flex-col gap-2 p-4 border border-secondary-dark rounded-lg">
		<h3 class="font-bold text-lg">1. Create Session</h3>
		<div>
			<label for="admin-key">Admin Key</label>
			<input type="password" id="admin-key" bind:value={adminKey} placeholder="Enter SWARM_API_KEY" />
		</div>
		<div>
			<label for="landscape-type">Landscape</label>
			<select id="landscape-type" bind:value={landscapeType}>
				<option value="quadratic">Quadratic (Easy)</option>
				<option value="ecological">Ecological (Medium)</option>
				<option value="rastrigin">Rastrigin (Hard)</option>
			</select>
		</div>
		<div>
			<label for="max-iterations">Max Iterations (Annealing)</label>
			<input type="number" id="max-iterations" bind:value={maxIterations} />
		</div>
		<button on:click={handleCreateSession} disabled={isLoading || !adminKey}>
			Create New Session
		</button>
	</div>

	<div class="flex flex-col gap-2 p-4 border border-secondary-dark rounded-lg">
		<h3 class="font-bold text-lg">2. Manage Session</h3>
		<div class="font-mono text-sm">Session ID: {$sessionState.id ?? 'N/A'}</div>
		<div class="font-mono text-sm">Status: {$sessionState.status ?? 'N/A'}</div>
		<div class="font-mono text-sm">Iteration: {$sessionState.iteration ?? 'N/A'}</div>
		<div class="grid grid-cols-3 gap-2 mt-2">
			<button on:click={() => handleAction('start')} disabled={isLoading || !$sessionState.id}>Start</button>
			<button on:click={() => handleAction('step')} disabled={isLoading || !$sessionState.id}>Step</button>
			<button on:click={() => handleAction('reset')} disabled={isLoading || !$sessionState.id} class="bg-yellow-600 hover:bg-yellow-700">Reset</button>
		</div>
	</div>
</div>
