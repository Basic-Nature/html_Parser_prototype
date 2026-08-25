# ElectionPulse Authentication Provider Roadmap

Status: architecture pickup note; not an active Keycloak integration.

## Current accepted authority

ElectionPulse currently supports progressive trust:

1. Anonymous access for public/read-only functionality.
2. Fresh ElectionPulse mTLS client-certificate proof.
3. A short-lived certificate-backed application session derived from that proof.

The current compatibility principal is `cert:<certificate DER SHA-256>`. It is
an internal credential identifier, not a claim about the user's civil identity.

## Provider-neutral contract

Application consumers should rely on the normalized authority vocabulary rather
than reconstructing identity from transport details:

- `anonymous`
- `fresh_certificate`
- `certificate_session`
- `federated_identity`
- `development_bypass`
- `authenticated_other`

Current provider: `electionpulse_mtls`.

Reserved future provider: `keycloak`.

Do not make public ElectionPulse functionality depend on Keycloak availability.

## Recommended Keycloak role

Use Keycloak as an optional identity provider/broker when ElectionPulse truly
needs durable person-level or organizational identity.

Preferred browser integration:

- OpenID Connect Authorization Code flow.
- PKCE where applicable.
- ElectionPulse receives normalized claims, then maps them into its own internal
  principal/authorization context.
- Do not implement Resource Owner Password/Direct Grant as the normal browser
  login path.
- Do not store a Keycloak user's password in ElectionPulse.
- Keep authorization scopes/roles explicit at the ElectionPulse boundary even
  when claims originate from Keycloak.

Information to obtain from the team before implementation:

- Keycloak base/issuer URL.
- Realm.
- Client ID.
- Whether the client is public or confidential.
- Approved redirect URIs.
- Logout/post-logout redirect expectations.
- Required scopes.
- Which claims identify the subject and groups/roles.
- Token/session lifetimes.
- Whether Keycloak brokers any upstream OIDC/SAML providers.
- Whether passkeys/WebAuthn are enabled.
- Whether X.509 authentication is already configured or desired.
- Dev/test realm details separate from production.

Never commit client secrets into the repository.

## Opaque ElectionPulse subjects

Target model:

- Device credential -> random opaque `dev_<id>`.
- Federated human subject `(issuer, sub)` -> random opaque `usr_<id>`.
- Service/capability credential -> random opaque `svc_<id>`.

The certificate fingerprint should remain server-side credential metadata after
migration rather than becoming the public/session identity.

A server-side random mapping is preferred over making the certificate
fingerprint itself the durable subject because it supports:

- certificate rotation without changing device identity;
- revocation;
- multiple credentials bound to one internal subject when explicitly allowed;
- reduced cross-system linkability;
- cleaner audit records.

The migration should preserve existing `cert:<fingerprint>` compatibility until
all consumers use the opaque subject contract.

## Capability tokens

Shared health/admin tokens are capabilities, not user identity.

If retained, prefer individually issued tokens with:

- explicit scope;
- expiration;
- revocation;
- rotation;
- audit label;
- no reuse as a human identity.

## Future step-up

Ordinary protected navigation may accept a bounded authenticated session.

High-risk operations may later require explicit fresh proof, for example:

- fresh mTLS certificate presentation;
- Keycloak step-up/ACR;
- WebAuthn/passkey reauthentication.

Fresh proof and session authority must remain separate concepts.
