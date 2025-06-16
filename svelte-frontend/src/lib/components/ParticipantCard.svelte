<script lang="ts">
	import type { Participant } from '$lib/stores/sessionStore';
	export let participant: Participant;
</script>

<div
	class="card flex flex-col gap-1 text-sm border-l-4"
	style="border-color: {participant.color ?? '#888'}"
>
	<div class="font-bold text-base">{participant.name}</div>
	<div>ID: <span class="font-mono text-xs">{participant.id}</span></div>
	{#if participant.position}
		<div>Position: [{participant.position.join(', ')}]</div>
	{/if}
	{#if participant.fitness !== undefined}
		<div>Fitness: {participant.fitness.toFixed(3)}</div>
	{/if}
	{#if participant.velocity_magnitude !== undefined}
		<div>Speed: {participant.velocity_magnitude.toFixed(3)}</div>
	{/if}
</div>
