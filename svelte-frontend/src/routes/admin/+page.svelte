<script lang="ts">
	import AdminPanel from '$lib/components/AdminPanel.svelte';
	import ParticipantList from '$lib/components/ParticipantList.svelte';
	import Grid from '$lib/components/Grid.svelte';
	import { eventLog, isAdminView } from '$lib/stores/sessionStore';
	import { onMount } from 'svelte';

	onMount(() => {
		isAdminView.set(true);
	});
</script>

<svelte:head>
	<title>Admin Dashboard | SwarmCraft</title>
</svelte:head>

<main class="grid lg:grid-cols-3 gap-6 p-6 h-screen">
	<div class="lg:col-span-1 flex flex-col gap-6">
		<AdminPanel />
		<div class="card flex-grow overflow-y-auto">
			<h3 class="text-xl font-bold mb-2">Event Log</h3>
			<div class="font-mono text-xs space-y-1">
                {#each $eventLog as log (log.id)}
					<div>
						<span class="text-gray-500">{log.timestamp}</span>
						<span class="ml-2">{log.message}</span>
					</div>
				{/each}
			</div>
		</div>
	</div>

	<div class="lg:col-span-1 flex flex-col gap-6">
		<h2 class="text-3xl font-bold text-center">Swarm Visualizer</h2>
		<Grid />
	</div>

	<div class="lg:col-span-1 flex flex-col">
		<ParticipantList />
	</div>
</main>
