-- W24 consolidated. Execute ONLY after separately approved PostgreSQL migration.
-- Target: existing Azure PostgreSQL Flexible Server, not a new database service.
-- Create/select a dedicated least-privileged role separately; no Redis.
BEGIN;
CREATE TABLE IF NOT EXISTS public.electionpulse_oidc_sessions (
    key_sha256 CHAR(64) PRIMARY KEY CHECK (key_sha256 ~ '^[0-9a-f]{64}$'),
    value_json TEXT NOT NULL CHECK (octet_length(value_json) BETWEEN 1 AND 2048),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS electionpulse_oidc_sessions_expiry_idx
    ON public.electionpulse_oidc_sessions (expires_at);
COMMIT;
-- After review, grant ONLY SELECT/INSERT/DELETE on this table to the dedicated
-- OIDC session role. The runtime must not have CREATE/ALTER or broader grants.
-- Do not execute this SQL by installing the consolidated package.
