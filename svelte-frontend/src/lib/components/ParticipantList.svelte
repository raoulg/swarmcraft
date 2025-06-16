<script lang="ts">
	import { sessionState } from '$lib/stores/sessionStore';
</script>

<div class="card h-full overflow-y-auto">
	<h3 class="text-xl font-bold mb-4">Participants ({$sessionState.participants.length})</h3>
	<div class="w-full text-sm text-left">
		<div class="bg-secondary-dark font-bold grid grid-cols-5 p-2 rounded-t-lg">
			<div>Name</div>
			<div>Position</div>
			<div>Fitness</div>
			<div>Speed</div>
			<div>Color</div>
		</div>
		<div class="max-h-[60vh] overflow-y-auto">
			{#each $sessionState.participants as p (p.id)}
				<div
					class="grid grid-cols-5 p-2 border-b border-secondary-dark hover:bg-secondary-dark/50"
				>
					<div>{p.name}</div>
					<div>{p.position ? `[${p.position.join(', ')}]` : 'N/A'}</div>
					<div>{p.fitness?.toFixed(2) ?? 'N/A'}</div>
					<div>{p.velocity_magnitude?.toFixed(2) ?? 'N/A'}</div>
					<div class="flex items-center gap-2">
						<div
							class="w-4 h-4 rounded-full"
							style="background-color: {p.color ?? '#888'}"
						/>
						<span class="font-mono text-xs">{p.color ?? ''}</span>
					</div>
				</div>
			{/each}
		</div>
	</div>
</div>
