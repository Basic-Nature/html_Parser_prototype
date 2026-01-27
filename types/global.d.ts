declare const bootstrap: any;
declare const io: any;
declare const socket: any;
declare function getActiveSessionId(): string | null;

/* ModalManager + placeholder ambient declarations (added by migration) */

interface ShowModalAction {
	id: string;
	label?: string;
	isDefault?: boolean;
	payload?: any;
}

interface ShowModalOptions {
	id?: string;
	title?: string;
	body?: string | Node | { html: string };
	actions?: ShowModalAction[];
	blocking?: boolean;
	priority?: number;
}

interface ModalResult {
	actionId: string;
	payload?: any;
	reason?: string;
}

interface ModalManager {
	container: HTMLElement | null;
	queue: Array<any>;
	active: any | null;
	showModal(opts: ShowModalOptions): Promise<ModalResult>;
	closeModal(id?: string, reason?: string): boolean;
	updateModal(id: string, partial: Partial<ShowModalOptions>): boolean;
	on(event: string, fn: (payload: any) => void): void;
	off(event: string, fn: (payload: any) => void): void;
	dump(): { active: { id: string; title: string } | null; queue: Array<{ id: string; title: string }> };
}

declare global {
	// Global shims for runtime libs used in JS files
	var io: any;
	var bootstrap: any;
	var socket: any;

	interface Window {
		modalManager?: ModalManager;
		// migration store for placeholder WeakMap (internal)
		__mm_placeholder_map?: WeakMap<Element, Comment>;
	}
	interface HTMLElement {
		/** Legacy placeholder property; migration prefers WeakMap */
		__mm_placeholder?: Comment | null;
	}
}

export {};
