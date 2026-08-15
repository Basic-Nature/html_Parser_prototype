---
layout: default
title: Architecture
---

# Election Pulse Architecture

This directory defines the durable boundaries and contracts of Election Pulse.

Architecture documentation describes how responsibilities are separated. It does
not claim that every target boundary has already been fully implemented.

## Core principle

Election Pulse separates:

- evidence from knowledge;
- parsing from presentation;
- orchestration from domain logic;
- runtime state from durable election records;
- source trust from data verification;
- safeguards from election-domain semantics.

The project has evolved from a command-line election parser into a web-accessible
election-data platform. The architecture preserves useful safeguards while
reducing accidental coupling between presentation, runtime state, parsing,
persistence, and learned context.

## Architecture documents

- [System overview](system_overview.md)
- [Parser pipeline](parser_pipeline.md)
- [Evidence model](evidence_model.md)
- [Context system](context_system.md)
- [Canonical election model](canonical_election_model.md)
- [Storage architecture](storage_architecture.md)
- [Automation and orchestration](automation.md)

## Authority

Source code is current implementation truth. CORE documents describe contracts
the repository is expected to implement. ARCHITECTURE describes durable
boundaries. Implementation history preserves earlier approaches without
remaining authoritative.

## Confidence authority

The boundary between evidence measurement and decision authority is defined in
[Confidence Authority](confidence_authority.md).

The central target ownership is:

- `risk_gates.py` - current normalized risk state;
- `risk_gates_calculus.py` - trajectory, boundary, and convergence;
- domain-specific components - evidence production and domain policy, not
  independent truth promotion.
