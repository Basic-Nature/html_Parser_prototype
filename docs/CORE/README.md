---
layout: default
title: Core Contracts
---

# Core Contracts

`docs/CORE` describes contracts the current implementation is expected to follow.

Architecture defines durable boundaries. CORE records concrete behavior that
code and tests should presently implement.

## Documents

- [Implemented contracts](implemented_contracts.md)
- [Constants reference](constants_reference.md)

## Authority

A CORE document should be testable against the repository.

If code no longer implements a documented contract, either restore the contract
or update CORE deliberately.

Do not place aspirational behavior in CORE.

## Architecture

See:

- [System overview](../ARCHITECTURE/system_overview.md)
- [Parser pipeline](../ARCHITECTURE/parser_pipeline.md)
- [Canonical election model](../ARCHITECTURE/canonical_election_model.md)
