/* Ambient declarations for runtime-injected globals used by webapp JS
   This file reduces TypeScript noise for window-scoped helpers injected
   by templates or other scripts. Keep minimal and add entries as needed. */

declare global {
  interface Window {
    openLeft?: () => void;
    openRight?: () => void;
    closeAll?: () => void;
    setOverlayVisible?: (v: boolean) => void;
    applyLogFilters?: () => void;
    socket?: any;
    bootstrap?: any;
    Chart?: any;
    __DATA_FRAMEWORK__?: { apiUrl?: string };
    __tl_helpers?: any;
    STATIC_ASSETS?: Record<string, string>;
    debugSocketIO?: any;
    __lastRunFlagged?: any;
    __lastRunReportPath?: string;
    clearSessionLogs?: () => void;
  }
}

export {};
