---
layout: default
title: Storage Architecture
---

# Storage Architecture

Election Pulse storage is organized by responsibility rather than file type.

A JSON file, database table, cache, or object store does not become authoritative
merely because it persists data.

## Storage classes

```text
source artifacts
evidence
canonical election data
learned context
runtime state
indexes / caches
telemetry / logs
generated reports
```

## Source artifacts

Source artifacts include downloaded election files, HTML, structured exports,
PDFs, screenshots, and uploads.

They should retain acquisition provenance and hashes where useful.

## Evidence storage

Evidence storage retains observations needed to explain parsing and review.

Evidence may be transient or durable depending on whether it supports canonical
output, discrepancy review, correction provenance, or audit reproduction.

## Canonical election storage

Canonical storage contains normalized, validated election records.

It should support precinct comparability, candidate/method completeness,
validation state, jurisdiction identity, and provenance relationships.

## Context storage

Context storage follows the categories in `context_system.md`:

- canonical;
- learned;
- runtime;
- indexes.

The category determines authority.

## Runtime state

Runtime state includes locks, migration checkpoints, telemetry, temporary parser
evidence, caches, PID files, and generated diagnostics.

Runtime state generally does not belong in Git.

## Generated indexes and LFS

Large generated artifacts may use Git LFS or external artifact storage.

The repository should avoid blanket LFS rules for all JSON/JSONL files. Large
generated files may be explicit LFS targets, while small mappings, schemas,
rules, and vocabularies remain reviewable in normal Git.

## Logs and telemetry

Logs are observability data, not an undocumented canonical knowledge store.

If a runtime observation is valuable enough to become learned context, it
should pass through promotion.

## Backups

Backup policy follows authority:

- runtime state may be regenerated;
- canonical and learned records require durable protection;
- source evidence required for audit may require retention;
- caches can generally be rebuilt.

A generic "backup every JSON file" strategy is not an architecture.

## Repository boundaries

Git is appropriate for source code, reviewed configuration, small canonical
vocabularies, documentation, tests, migrations, and explicit fixtures.

Git is generally inappropriate for runtime telemetry, locks, temporary output,
local databases, caches, transient OCR output, and session-specific evidence.

## Cost-aware deployment

Production deployment should not carry repository artifacts or generated data
that runtime execution does not require.

Pruning redundant files and separating bulky artifacts can reduce Azure runtime
and storage cost without weakening auditability.
