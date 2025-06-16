<script lang="ts">
	import { sessionState } from '$lib/stores/sessionStore';

	let gridSize = 25;
	let particles = [];

	sessionState.subscribe((s) => {
		gridSize = s.config?.grid_size ?? 25;
		particles = s.participants ?? [];
	});
</script>

<div class="aspect-square w-full max-w-[70vh] relative bg-primary-dark rounded-xl shadow-inner">
		<div
		class="grid w-full h-full"
		style="grid-template-columns: repeat({gridSize}, 1fr); grid-template-rows: repeat({gridSize}, 1fr);"
		>
			{#each Array(gridSize * gridSize) as _, i (i)}
				<div class="border border-secondary-dark/50"></div>
			{/each}
		</div>

	{#each particles as particle (particle.id)}
		{#if particle.position}
			<div
				class="particle absolute rounded-full transition-all duration-500 ease-in-out flex items-center justify-center text-xs font-bold text-white/70"
				style="
                    width: calc(100% / {gridSize});
                    height: calc(100% / {gridSize});
                    top: {particle.position[0] * (100 / gridSize)}%;
                    left: {particle.position[1] * (100 / gridSize)}%;
                    background-color: {particle.color ?? '#888'};
                    box-shadow: 0 0 10px {particle.color ?? '#888'};
                "
				title={particle.name}
			>
				{particle.name.substring(0, 1).toUpperCase()}
			</div>
		{/if}
	{/each}
</div>
