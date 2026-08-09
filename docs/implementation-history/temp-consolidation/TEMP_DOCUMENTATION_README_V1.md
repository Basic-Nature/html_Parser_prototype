# Temporary Documentation

This folder is **gitignored** and used for:

- Working drafts and session notes
- Experimental design documents
- Temporary collaboration files (developer + Copilot)
- Files that should not clutter the main docs/ structure

Files in this folder are **NOT committed to GitHub** and are for local development only.

## Usage

Create temporary documentation here instead of at root:

```bash
docs/temp/session_notes_2026_02_19.md
docs/temp/draft_feature_spec.md
docs/temp/working_architecture_ideas.md
```

## Moving to Production Docs

When a document is finalized, move it to the appropriate docs/ subfolder:

- `docs/FEATURES/` - Feature documentation
- `docs/CORE/` - Core architecture
- `docs/DEVELOPMENT/` - Development guides
- `docs/DEPLOYMENT/` - Deployment guides
- `docs/session-logs/` - Session summaries (if worth preserving)
