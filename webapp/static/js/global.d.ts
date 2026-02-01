export {};

declare global {
  interface Window {
    // Programmatic helper to manage overlay focus/inert state for modals and dropdowns
    manageOverlayFocus?: (selector: string, open: boolean) => void;

    // Optional capture-only helper that returns a promise when the nav dropdown is open
    openNavDropdownForCapture?: () => Promise<void>;
  }

  // Augment Element/HTMLElement to include runtime properties used by our helpers
  interface Element {
    // Backup storage used by manageOverlayFocus for restoring attributes
    __manage_backup?: any;
    // Some browsers expose 'inert' on HTMLElement; declare it to avoid TS errors
    inert?: boolean;
    // focus may be invoked on elements in our code paths; accept an optional options arg
    focus?: (options?: FocusOptions) => void;
  }

  interface HTMLElement {
    inert?: boolean;
    focus?: (options?: FocusOptions) => void;
  }

  /** Minimal FocusOptions for focus() calls used in the codebase. */
  interface FocusOptions {
    preventScroll?: boolean;
  }
}
