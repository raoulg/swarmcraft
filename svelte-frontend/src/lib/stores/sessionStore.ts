import { writable } from 'svelte/store';

// Define types for our state
export interface Participant {
	id: string;
	name: string;
	position?: [number, number];
	fitness?: number;
	velocity_magnitude?: number;
	color?: string;
	role?: string; // For future use (GWO, ABC)
}

export interface SessionConfig {
	grid_size: number;
	landscape_type: string;
}

export interface Session {
	id?: string;
	code?: string;
	status?: string;
	config?: SessionConfig;
	participants: Participant[];
	iteration?: number;
}
export interface LogEntry {
	id: string;
	timestamp: string;
	message: string;
	data?: unknown;
}

// Create stores
export const sessionState = writable<Session>({ participants: [] });
export const participantId = writable<string | null>(null);
export const eventLog = writable<LogEntry[]>([]);
export const isAdminView = writable<boolean>(false);

// Helper to add logs
export function logEvent(message: string, data?: unknown) {
	const newLog: LogEntry = {
		id: crypto.randomUUID(),
		timestamp: new Date().toLocaleTimeString(),
		message,
		data
	};
	eventLog.update((logs) => [newLog, ...logs.slice(0, 100)]);
}
