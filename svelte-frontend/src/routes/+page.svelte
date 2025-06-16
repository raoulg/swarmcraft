<script lang="ts">
	import { joinSession } from '$lib/api/client';
	import { sessionState, participantId } from '$lib/stores/sessionStore';
	import Grid from '$lib/components/Grid.svelte';
	import ParticipantCard from '$lib/components/ParticipantCard.svelte';
	import { slide } from 'svelte/transition';

	let sessionCode = '';
	let errorMessage = '';
	let isLoading = false;
	let myParticipantDetails: Participant | undefined;

	// Update local participant details when the main store changes
	sessionState.subscribe(s => {
		if ($participantId) {
			myParticipantDetails = s.participants.find(p => p.id === $participantId);
		}
	});

	async function handleJoin() {
		isLoading = true;
		errorMessage = '';
		try {
			await joinSession(sessionCode.toUpperCase());
		} catch (e) {
			errorMessage = e.message;
		}
		isLoading = false;
	}
</script>

<svelte:head>
	<title>SwarmCraft</title>
</svelte:head>

<main class="p-4 md:p-8 max-w-7xl mx-auto">
	<header class="text-center mb-8">
		<h1 class="text-5xl font-extrabold text-accent-color tracking-wider">SwarmCraft</h1>
		<p class="text-lg text-text-color/80 mt-2">Interactive Swarm Intelligence</p>
	</header>

	{#if !$sessionState.id}
		<div transition:slide class="max-w-md mx-auto card flex flex-col gap-4">
			<h2 class="text-2xl font-bold text-center">Join a Session</h2>
			<input
				type="text"
				bind:value={sessionCode}
				placeholder="Enter Session Code"
				class="text-center text-xl tracking-widest"
				maxlength="6"
				on:input={(e) => (e.currentTarget.value = e.currentTarget.value.toUpperCase())}
			/>
			<button on:click={handleJoin} disabled={isLoading || sessionCode.length < 6}>
				{#if isLoading}Joining...{:else}Join{/if}
			</button>
			{#if errorMessage}
				<p class="text-red-400 text-center">{errorMessage}</p>
			{/if}
		</div>
	{:else}
		<div transition:slide class="grid md:grid-cols-3 gap-8">
			<div class="md:col-span-2">
				<Grid />
			</div>

			<div class="flex flex-col gap-4">
				<div class="card">
					<h3 class="text-xl font-bold mb-2">Session Info</h3>
					<p>Status: <span class="font-bold text-accent-color">{$sessionState.status}</span></p>
					<p>Iteration: <span class="font-bold">{$sessionState.iteration ?? 0}</span></p>
				</div>
				{#if myParticipantDetails}
				<div class="card">
					<h3 class="text-xl font-bold mb-2">Your Stats</h3>
					<ParticipantCard participant={myParticipantDetails} />
				</div>
				{/if}
				<div class="card">
					<h3 class="text-xl font-bold mb-2">Other Participants</h3>
				<div class="flex flex-wrap gap-2">
					{#each $sessionState.participants as p (p.id)}
						{#if p.id !== $participantId}
							<div
								class="w-8 h-8 rounded-full"
								title={p.name}
								style="background-color: {p.color ?? '#888'};"
							></div>
						{/if}
					{/each}
				</div>

				</div>
			</div>
		</div>
	{/if}
</main>
