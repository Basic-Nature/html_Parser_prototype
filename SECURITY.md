# Security Policy

Election Pulse handles election data, parser evidence, authentication flows, uploaded files, database connections, and operational diagnostics.

Security contributions and vulnerability reports are taken seriously.

This document explains how to report security issues, which systems are in scope, and the security principles contributors should follow.

---

## Reporting a Vulnerability

Do not open a public GitHub issue for a vulnerability that could expose:

* credentials
* private keys
* access tokens
* client certificates
* authentication bypasses
* database access
* uploaded files
* private election data
* administrative endpoints
* command execution
* path traversal
* arbitrary file access
* sensitive logs
* deployment secrets

Instead, report the issue privately to the project maintainers.

Include as much of the following as possible:

* affected component
* affected route, file, or workflow
* vulnerability type
* steps to reproduce
* expected behavior
* observed behavior
* potential impact
* environment where it occurred
* relevant logs or screenshots
* suggested mitigation, if known

Do not include active credentials, private keys, production certificates, or unnecessary personal data in the report.

If no private reporting channel is currently configured in the repository, contact the repository owner directly through GitHub and request a private channel before sharing exploit details.

---

## Responsible Disclosure

Please provide maintainers a reasonable opportunity to investigate and address a reported vulnerability before publishing technical details.

Do not:

* access data that does not belong to you
* modify election data
* disrupt production services
* perform denial-of-service testing
* attempt privilege escalation beyond what is needed to demonstrate the issue
* retain downloaded sensitive information
* publicly disclose an unpatched vulnerability without coordination
* test against third-party election systems without authorization

Security research should remain limited to systems and environments you are authorized to test.

---

## Supported Versions

Election Pulse is under active development.

Security fixes are generally applied to the current default branch.

Older commits, archived branches, historical documentation, prototype code, and superseded deployment configurations may not receive security updates.

Before reporting an issue against historical code, confirm whether the behavior still exists in the current branch.

---

## Security Scope

Security-sensitive areas include:

```text
Authentication and authorization
Certificate handling
Session management
File uploads
Archive extraction
Path validation
Parser execution
Administrative routes
Health and automation endpoints
Database access
Environment configuration
Logging and diagnostics
Content Security Policy
Socket.IO communication
Generated downloads
Cloud deployment workflows
```

A weakness in any of these areas may affect the confidentiality, integrity, or availability of Election Pulse.

---

## Authentication and Authorization

Election Pulse may use environment-dependent authentication behavior.

Production and public deployments must not assume that local-development authentication behavior is appropriate for deployment.

Security-sensitive routes should explicitly enforce authorization rather than relying only on:

* hidden navigation
* frontend controls
* URL obscurity
* deployment assumptions
* client-side checks

Authentication status and authorization decisions should be evaluated server-side.

Administrative, mutation, ingestion, health, and diagnostic endpoints should use explicit access controls appropriate to their risk.

---

## Client Certificate Handling

Certificate-aware deployments must treat certificate metadata as untrusted until validated by the trusted deployment boundary.

Do not trust arbitrary client-supplied headers that claim to contain certificate information.

Certificate metadata forwarded by a reverse proxy or cloud platform should only be accepted when:

* the forwarding infrastructure is trusted
* direct client access cannot spoof the trusted headers
* the expected certificate fields are validated
* authorization decisions use verified certificate state
* sensitive certificate data is not exposed unnecessarily

The application should distinguish between:

```text
Certificate present
Certificate parsed
Certificate validated
Identity authenticated
Action authorized
```

These states are not interchangeable.

---

## Secrets and Environment Variables

Never commit populated environment files or live secrets.

Examples include:

```text
.env
database passwords
API keys
private keys
client certificates
service-account credentials
OAuth secrets
access tokens
session secrets
cloud deployment credentials
```

Use the repository's environment template only for variable names, placeholders, and documentation.

The environment template must never contain live values.

Production secrets should be stored using the deployment platform's protected configuration or secret-management system.

If a secret is accidentally committed:

1. Treat it as compromised.
2. Revoke or rotate it immediately.
3. Remove it from active configuration.
4. Review logs for possible misuse.
5. Remove it from repository history where practical.
6. Document the incident privately.

Deleting the file in a later commit does not invalidate a leaked secret.

---

## File Upload Security

Uploaded files must be treated as untrusted.

Upload handling should include:

* filename sanitization
* path normalization
* extension validation
* content validation where practical
* maximum file-size enforcement
* archive traversal protection
* controlled storage locations
* access-control enforcement
* temporary-file cleanup
* explicit parser selection
* safe error handling

Do not trust a file solely because its extension appears valid.

For example, a file named:

```text
results.pdf
```

may not contain valid PDF content.

Uploaded archives must be checked for paths such as:

```text
../../sensitive-file
```

before extraction.

---

## Path Safety

All filesystem paths derived from users, URLs, filenames, form data, database values, or uploaded content must be validated.

Security-sensitive code should prevent:

* directory traversal
* absolute-path injection
* unintended symbolic-link traversal
* overwriting protected files
* reading outside approved directories
* writing into source-controlled locations unintentionally

Prefer resolved `pathlib.Path` operations and explicit allowed root directories.

A string beginning with an expected folder name is not sufficient proof that the resolved path remains inside that folder.

---

## Parser Execution Safety

Parser inputs may contain malformed, hostile, or unexpectedly large data.

Parser code should account for:

* excessive file size
* deeply nested structures
* malformed HTML
* malformed JSON
* corrupted PDFs
* decompression bombs
* unexpectedly large tables
* long-running OCR tasks
* browser automation hangs
* untrusted redirects
* external resource requests
* memory exhaustion
* recursive parsing behavior

Long-running operations should support timeouts, cancellation, resource cleanup, and bounded concurrency where practical.

Parser failures should not expose secrets, internal paths, or sensitive environment details to unauthenticated users.

---

## Browser Automation

Browser automation should be treated as execution against untrusted remote content.

Contributors should avoid:

* disabling browser security controls unnecessarily
* executing arbitrary page-provided scripts outside the browser sandbox
* exposing local files to browser contexts
* persisting authenticated browser profiles in source control
* logging session tokens or cookies
* downloading files into unrestricted locations
* following unvalidated redirects into internal services

Browser contexts should be isolated per task where practical.

Downloaded files should pass through the same validation and storage controls as directly uploaded files.

---

## Server-Side Request Forgery

Features that retrieve URLs must protect against server-side request forgery.

URL validation should consider:

* localhost
* loopback addresses
* private network ranges
* link-local addresses
* cloud metadata endpoints
* unusual URL schemes
* redirects into restricted networks
* DNS rebinding
* encoded IP-address representations

Public URL ingestion should not provide unrestricted access to internal network resources.

Any intentional internal-network support should be explicitly configured and narrowly scoped.

---

## Database Security

Database access should use least privilege.

Application credentials should only have the permissions required for their function.

Contributors should:

* use parameterized queries
* avoid constructing SQL from untrusted strings
* separate migration privileges from application privileges where practical
* validate imported data before persistence
* preserve batch and provenance metadata
* avoid exposing database errors directly to users
* protect connection strings
* review destructive migration operations carefully

Local SQLite or development databases must not be confused with production configuration.

Production database selection should be explicit and verifiable.

---

## Evidence and Data Integrity

Election Pulse distinguishes source evidence from normalized and canonical data.

Security and integrity protections should preserve that separation.

A parser, automation task, ML process, or user-facing workflow must not silently:

* rewrite source evidence
* replace missing values with invented values
* promote unreviewed observations into canonical knowledge
* remove candidates with zero votes
* discard vote methods
* hide reconciliation failures
* overwrite original provenance
* suppress discrepancies

Changes to election data should remain attributable and reproducible.

---

## Logging and Diagnostics

Logs should support troubleshooting and auditability without exposing sensitive information.

Do not log:

* passwords
* API keys
* private keys
* access tokens
* session cookies
* complete certificate material
* database connection strings containing credentials
* unnecessary personal data
* raw authorization headers

Be cautious when logging:

* local filesystem paths
* uploaded filenames
* source URLs containing query secrets
* certificate metadata
* user identifiers
* database errors
* request bodies

Diagnostic output generated during local development should normally remain untracked.

Runtime logs, telemetry, PID files, process locks, OCR output, temporary evidence, and debug reports should not be committed unless they are intentionally curated fixtures.

---

## Content Security Policy

Election Pulse uses Content Security Policy controls to reduce the risk of script injection and unsafe browser behavior.

Frontend contributions should avoid:

* inline JavaScript
* inline CSS where external assets are required
* `eval`
* dynamically constructed executable code
* untrusted HTML insertion
* unsafe event-handler attributes
* unnecessary third-party scripts
* weakening CSP directives to solve local implementation problems

Use external JavaScript and CSS assets.

When dynamic content must be rendered, prefer safe DOM APIs and explicit escaping or sanitization.

Changes that require CSP relaxation should include a clear security justification.

---

## Cross-Site Scripting

All untrusted values rendered into HTML must be escaped or safely sanitized.

Particular care is required for:

* parser output
* uploaded filenames
* election contest names
* candidate names
* source URLs
* error messages
* logs displayed in the browser
* OCR text
* administrative diagnostics

Do not use unsafe HTML insertion merely to preserve formatting.

Text that originated from an election source remains untrusted even when the source is an official public website.

---

## Cross-Site Request Forgery

State-changing routes should include appropriate CSRF protections or equivalent safeguards.

Sensitive actions should not rely solely on the request being difficult to guess.

Examples include:

* deleting files
* changing parser configuration
* promoting learned context
* starting health tasks
* changing authentication settings
* importing data
* initiating administrative operations

API authentication, same-site cookies, origin checks, CSRF tokens, and explicit authorization should be applied according to the route and deployment model.

---

## Session Security

Session configuration should use secure settings appropriate to the deployment environment.

Production deployments should consider:

* strong secret keys
* secure cookies
* HTTP-only cookies
* appropriate SameSite behavior
* HTTPS enforcement
* session expiration
* session invalidation
* protection against fixation
* limited sensitive data in session storage

Development defaults must not silently become production defaults.

---

## Socket.IO and Real-Time Events

Real-time connections must follow the same authentication and authorization expectations as ordinary HTTP routes.

Do not assume that a user authorized to load a page is automatically authorized for every Socket.IO event.

Event handlers should validate:

* authenticated identity
* authorization
* session state
* payload structure
* task ownership
* target resources
* message size

Do not emit sensitive logs, parser data, or administrative results to unauthorized clients.

Multi-worker deployments must use a supported message-queue configuration when required for consistent event delivery.

---

## Administrative and Health Endpoints

Health, observability, diagnostic, and automation endpoints may expose operational details or trigger sensitive actions.

These endpoints should be divided conceptually into:

```text
Public liveness
Authenticated diagnostics
Privileged administration
Mutation or task execution
```

A simple liveness check should expose minimal information.

Detailed environment, dependency, database, certificate, file, parser, or deployment information should require appropriate authorization.

Endpoints capable of running scripts, modifying files, changing context, or starting parser tasks require stronger protections.

---

## Automation and CI/CD

Automation workflows should follow least privilege.

Workflow changes should be reviewed for:

* secret exposure
* unsafe pull-request execution
* untrusted artifact handling
* command injection
* excessive repository permissions
* deployment-environment access
* branch protection bypass
* dependency compromise
* artifact retention
* insecure debugging output

Avoid printing secrets or environment variables in workflow logs.

Pin actions and dependencies where appropriate.

Deployment workflows should fail clearly when required files or configuration are missing rather than silently substituting unexpected behavior.

---

## Dependency Security

Dependencies should be added deliberately.

Before introducing a dependency, consider:

* whether the capability already exists in the project
* maintenance activity
* release history
* transitive dependencies
* license compatibility
* known vulnerabilities
* runtime privilege
* package size
* deployment impact

Avoid introducing a large dependency to solve a small problem without justification.

Security updates should be tested against parser, UI, OCR, ML, and deployment compatibility before broad rollout.

---

## Machine Learning and NLP Safety

ML and NLP output must not be treated as authoritative solely because a model produced it.

Model-assisted behavior should preserve:

* input evidence
* model or rule identity
* version
* confidence
* provenance
* validation status
* review status

Unreviewed model output must not silently become canonical election knowledge.

Training data should not include secrets, unnecessary personal information, or unapproved parser evidence.

---

## Public Election Sources

Election Pulse interacts with public election systems.

Contributors must respect:

* applicable law
* site terms
* rate limits
* access controls
* robots and usage policies where applicable
* authorized research boundaries

Do not attempt to bypass security controls on third-party election systems.

CAPTCHA handling should support legitimate human interaction and resilient access workflows without becoming a general-purpose security bypass system.

---

## Security Testing

Security-relevant changes should include appropriate validation.

Possible checks include:

* authentication tests
* authorization tests
* path traversal tests
* upload validation tests
* CSP tests
* session tests
* route contract tests
* URL validation tests
* SSRF tests
* database input tests
* secret-scanning checks
* dependency audits
* static analysis

Before submitting security-sensitive changes, run the smallest relevant test suite and then broader validation where practical.

Useful commands may include:

```bash
python -m compileall webapp
python -m pytest
git diff --check
npm run check-js
npm run lint
```

Repository automation may provide additional validation stages.

Do not report a test suite as passing when it was skipped, interrupted, or blocked by missing dependencies.

---

## Security Review Checklist

Before merging a security-sensitive contribution, confirm:

* no secrets are included
* authentication is server-side
* authorization is explicit
* untrusted paths are resolved safely
* uploaded content is validated
* external URLs are restricted appropriately
* database queries are parameterized
* sensitive logs are redacted
* frontend output is escaped
* CSP requirements remain intact
* state-changing routes are protected
* Socket.IO events enforce authorization
* runtime files remain untracked
* evidence remains distinct from canonical knowledge
* failures remain visible
* relevant tests exist
* deployment assumptions are documented

---

## Known Limitations

Election Pulse is under active development and architectural consolidation.

Some older modules, scripts, documentation, or workflows may not yet fully reflect the current security model.

A historical security claim should not be assumed valid without checking the current implementation.

When encountering conflicting behavior:

1. Prefer the current code over historical documentation.
2. Prefer explicit server-side enforcement over comments.
3. Report unclear security ownership.
4. Avoid extending legacy behavior without review.
5. Add regression tests when correcting the issue.

---

## Security Improvements

Security improvements are welcome in areas including:

* authentication
* certificate validation
* route authorization
* upload controls
* path validation
* URL safety
* CSP
* frontend sanitization
* secret management
* database access
* dependency hygiene
* workflow permissions
* audit logging
* test coverage
* deployment verification

Security refactors should remain focused and should not silently alter election-data semantics.

---

## Final Principle

Election Pulse is designed around traceability and trust.

Security supports that goal by protecting:

```text
The source
The evidence
The transformation
The stored data
The review process
The final result
```

When choosing between convenience and a clear, auditable security boundary, prefer the auditable boundary.
