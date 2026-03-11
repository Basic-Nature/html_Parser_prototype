from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from webapp.parser.config import PROJECT_ROOT


def _iter_vocab_lines(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                yield line
    except Exception:
        return


def _normalize(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _normalize_loose(value: str) -> str:
    value = _normalize(value)
    if not value:
        return ""
    value = value.replace("&", " and ")
    value = value.replace("/", " ")
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _parse_mapping_line(line: str) -> tuple[str, str] | None:
    for sep in ("->", "=>", "=", ":"):
        if sep in line:
            left, right = line.split(sep, 1)
            alias = left.strip()
            target = right.strip()
            if alias and target:
                return alias, target
    return None


_RULE_FILE_MARKERS = (
    "regex",
    "resolver",
    "selector",
    "field_type",
    "canonical_segment",
)


def _file_category(file_name: str) -> str:
    lowered = (file_name or "").lower()
    if any(marker in lowered for marker in _RULE_FILE_MARKERS):
        return "rule_directive"
    return "entity_alias"


def _is_rule_like_mapping(alias_raw: str, target_raw: str) -> bool:
    alias = alias_raw or ""
    target = target_raw or ""
    target_lower = target.lower()
    if "|" in alias or "\\b" in alias.lower() or "^" in alias or "$" in alias:
        return True
    if any(token in target_lower for token in ("when=", "tie_break=", "priority", "context_required")):
        return True
    if "," in target and " " not in target:
        return True
    return False


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _score_candidate(alias_loose: str, target_loose: str, candidate_loose: str) -> float:
    alias_sim = _similarity(alias_loose, candidate_loose)
    target_sim = _similarity(target_loose, candidate_loose)
    if target_loose:
        base = (0.7 * target_sim) + (0.3 * alias_sim)
    else:
        base = max(alias_sim, target_sim)
    bonus = 0.0

    if target_loose and target_loose in candidate_loose:
        bonus += 0.1
    if alias_loose and alias_loose in candidate_loose:
        bonus += 0.1

    alias_tokens = [tok for tok in alias_loose.split() if tok]
    candidate_tokens = [tok for tok in candidate_loose.split() if tok]
    if alias_tokens and candidate_tokens:
        first_alias = alias_tokens[0]
        first_candidate = candidate_tokens[0]
        if len(first_alias) <= 4 and first_candidate.startswith(first_alias):
            bonus += 0.12

    if candidate_loose.endswith("."):
        bonus -= 0.05

    return min(1.0, base + bonus)


def _collect_entity_values(vocab_root: Path) -> tuple[set[str], set[str], list[tuple[str, str]]]:
    entities_dir = vocab_root / "entities"
    entity_values: set[str] = set()
    entity_values_loose: set[str] = set()
    canonicals: list[tuple[str, str]] = []

    for path in sorted(entities_dir.glob("*.txt")):
        for line in _iter_vocab_lines(path):
            normalized = _normalize(line)
            if not normalized:
                continue
            loose = _normalize_loose(line)
            entity_values.add(normalized)
            if loose:
                entity_values_loose.add(loose)
                canonicals.append((line, loose))

    return entity_values, entity_values_loose, canonicals


def get_vocab_alignment_report(*, sample_limit: int = 25) -> dict[str, Any]:
    """Audit alias->canonical mapping alignment against entity vocab files."""
    sample_limit = max(1, min(int(sample_limit or 25), 200))

    vocab_root = PROJECT_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab"
    validators_dir = vocab_root / "validators"

    entity_values, entity_values_loose, _canonicals = _collect_entity_values(vocab_root)
    entity_file_counts: dict[str, int] = {}
    for path in sorted((vocab_root / "entities").glob("*.txt")):
        count = 0
        for line in _iter_vocab_lines(path):
            norm = _normalize(line)
            if norm:
                count += 1
        entity_file_counts[path.name] = count

    per_file: dict[str, dict[str, Any]] = {}
    unresolved_samples: list[dict[str, str]] = []
    unresolved_entity_samples: list[dict[str, str]] = []
    total_mappings = 0
    resolved_mappings = 0
    resolved_exact = 0
    resolved_normalized = 0
    category_summary = {
        "entity_alias": {"files": 0, "mapping_count": 0, "resolved": 0, "resolved_exact": 0, "resolved_normalized": 0},
        "rule_directive": {"files": 0, "mapping_count": 0, "resolved": 0, "resolved_exact": 0, "resolved_normalized": 0},
    }

    for path in sorted(validators_dir.glob("*.txt")):
        file_category = _file_category(path.name)
        category_summary[file_category]["files"] += 1

        mappings = 0
        resolved = 0
        unresolved = 0
        file_exact = 0
        file_normalized = 0
        file_entity_alias = 0
        file_rule_directive = 0

        for line in _iter_vocab_lines(path):
            parsed = _parse_mapping_line(line)
            if not parsed:
                continue
            alias_raw, target_raw = parsed

            mapping_category = file_category
            if mapping_category == "entity_alias" and _is_rule_like_mapping(alias_raw, target_raw):
                mapping_category = "rule_directive"

            mappings += 1
            total_mappings += 1
            category_summary[mapping_category]["mapping_count"] += 1
            if mapping_category == "entity_alias":
                file_entity_alias += 1
            else:
                file_rule_directive += 1

            alias = _normalize(alias_raw)
            target = _normalize(target_raw)
            alias_loose = _normalize_loose(alias_raw)
            target_loose = _normalize_loose(target_raw)

            mode = "unresolved"
            if target in entity_values or target == alias:
                mode = "exact"
            elif target_loose and (target_loose in entity_values_loose or target_loose == alias_loose):
                mode = "normalized"

            is_resolved = mode != "unresolved"
            if is_resolved:
                resolved += 1
                resolved_mappings += 1
                category_summary[mapping_category]["resolved"] += 1
                if mode == "exact":
                    file_exact += 1
                    resolved_exact += 1
                    category_summary[mapping_category]["resolved_exact"] += 1
                elif mode == "normalized":
                    file_normalized += 1
                    resolved_normalized += 1
                    category_summary[mapping_category]["resolved_normalized"] += 1
            else:
                unresolved += 1
                if len(unresolved_samples) < sample_limit:
                    unresolved_samples.append(
                        {
                            "file": path.name,
                            "alias": alias_raw,
                            "target": target_raw,
                        }
                    )
                if mapping_category == "entity_alias" and len(unresolved_entity_samples) < sample_limit:
                    unresolved_entity_samples.append(
                        {
                            "file": path.name,
                            "alias": alias_raw,
                            "target": target_raw,
                        }
                    )

        per_file[path.name] = {
            "category": file_category,
            "mappings": mappings,
            "entity_alias_mappings": file_entity_alias,
            "rule_directive_mappings": file_rule_directive,
            "resolved": resolved,
            "unresolved": unresolved,
            "resolved_exact": file_exact,
            "resolved_normalized": file_normalized,
            "resolution_rate": round((resolved / mappings) * 100, 2) if mappings else None,
        }

    unresolved_total = max(0, total_mappings - resolved_mappings)
    resolution_rate = round((resolved_mappings / total_mappings) * 100, 2) if total_mappings else None

    entity_alias_mapping_count = int(category_summary["entity_alias"]["mapping_count"])
    entity_alias_resolved = int(category_summary["entity_alias"]["resolved"])
    entity_alias_unresolved = max(0, entity_alias_mapping_count - entity_alias_resolved)
    entity_alias_rate = round((entity_alias_resolved / entity_alias_mapping_count) * 100, 2) if entity_alias_mapping_count else None

    rule_mapping_count = int(category_summary["rule_directive"]["mapping_count"])
    rule_resolved = int(category_summary["rule_directive"]["resolved"])
    rule_unresolved = max(0, rule_mapping_count - rule_resolved)
    rule_rate = round((rule_resolved / rule_mapping_count) * 100, 2) if rule_mapping_count else None

    return {
        "vocab_root": str(vocab_root),
        "entities": {
            "files": entity_file_counts,
            "file_count": len(entity_file_counts),
            "entry_count": len(entity_values),
        },
        "validators": {
            "file_count": len(per_file),
            "mapping_count": total_mappings,
            "resolved_count": resolved_mappings,
            "unresolved_count": unresolved_total,
            "resolved_exact": resolved_exact,
            "resolved_normalized": resolved_normalized,
            "resolution_rate": resolution_rate,
            "files": per_file,
            "categories": {
                "entity_alias": {
                    "file_count": int(category_summary["entity_alias"]["files"]),
                    "mapping_count": entity_alias_mapping_count,
                    "resolved_count": entity_alias_resolved,
                    "unresolved_count": entity_alias_unresolved,
                    "resolved_exact": int(category_summary["entity_alias"]["resolved_exact"]),
                    "resolved_normalized": int(category_summary["entity_alias"]["resolved_normalized"]),
                    "resolution_rate": entity_alias_rate,
                },
                "rule_directive": {
                    "file_count": int(category_summary["rule_directive"]["files"]),
                    "mapping_count": rule_mapping_count,
                    "resolved_count": rule_resolved,
                    "unresolved_count": rule_unresolved,
                    "resolved_exact": int(category_summary["rule_directive"]["resolved_exact"]),
                    "resolved_normalized": int(category_summary["rule_directive"]["resolved_normalized"]),
                    "resolution_rate": rule_rate,
                },
            },
        },
        "entity_only": {
            "mapping_count": entity_alias_mapping_count,
            "resolved_count": entity_alias_resolved,
            "unresolved_count": entity_alias_unresolved,
            "resolved_exact": int(category_summary["entity_alias"]["resolved_exact"]),
            "resolved_normalized": int(category_summary["entity_alias"]["resolved_normalized"]),
            "resolution_rate": entity_alias_rate,
        },
        "samples": {
            "unresolved": unresolved_samples,
            "unresolved_entity_only": unresolved_entity_samples,
        },
    }


def get_vocab_alignment_suggestions(*, limit: int = 50, min_score: float = 0.45) -> dict[str, Any]:
    """Suggest canonical entity targets for unresolved alias mappings."""
    limit = max(1, min(int(limit or 50), 200))
    min_score = max(0.0, min(float(min_score), 0.99))

    vocab_root = PROJECT_ROOT / "webapp" / "parser" / "Context_Integration" / "vocab"
    validators_dir = vocab_root / "validators"
    entity_values, entity_values_loose, canonicals_raw = _collect_entity_values(vocab_root)

    canonical_by_loose: dict[str, str] = {}
    for canonical_raw, canonical_loose in canonicals_raw:
        if canonical_loose and canonical_loose not in canonical_by_loose:
            canonical_by_loose[canonical_loose] = canonical_raw
    canonicals = [(raw, loose) for loose, raw in canonical_by_loose.items()]

    by_first_char: dict[str, list[tuple[str, str]]] = {}
    by_first_token: dict[str, list[tuple[str, str]]] = {}
    for canonical_raw, canonical_loose in canonicals:
        if canonical_loose:
            first_char = canonical_loose[0]
            by_first_char.setdefault(first_char, []).append((canonical_raw, canonical_loose))
            first_token = canonical_loose.split()[0]
            if first_token:
                by_first_token.setdefault(first_token, []).append((canonical_raw, canonical_loose))

    unresolved: list[dict[str, str]] = []
    for path in sorted(validators_dir.glob("*.txt")):
        file_category = _file_category(path.name)
        for line in _iter_vocab_lines(path):
            parsed = _parse_mapping_line(line)
            if not parsed:
                continue
            alias_raw, target_raw = parsed

            mapping_category = file_category
            if mapping_category == "entity_alias" and _is_rule_like_mapping(alias_raw, target_raw):
                mapping_category = "rule_directive"
            if mapping_category != "entity_alias":
                continue

            alias_norm = _normalize(alias_raw)
            target_norm = _normalize(target_raw)
            alias_loose = _normalize_loose(alias_raw)
            target_loose = _normalize_loose(target_raw)

            resolved = False
            if target_norm in entity_values or target_norm == alias_norm:
                resolved = True
            elif target_loose and (target_loose in entity_values_loose or target_loose == alias_loose):
                resolved = True

            if not resolved:
                unresolved.append(
                    {
                        "file": path.name,
                        "alias": alias_raw,
                        "target": target_raw,
                        "alias_loose": alias_loose,
                        "target_loose": target_loose,
                    }
                )

    deduped: list[dict[str, str]] = []
    seen = set()
    for item in unresolved:
        key = (item.get("file"), item.get("alias"), item.get("target"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    suggestions: list[dict[str, Any]] = []

    def _candidate_pool(alias_loose: str, target_loose: str) -> list[tuple[str, str]]:
        pool: list[tuple[str, str]] = []
        seen = set()

        for probe in (target_loose, alias_loose):
            if not probe:
                continue
            first_char = probe[0]
            for row in by_first_char.get(first_char, []):
                key = row[1]
                if key not in seen:
                    seen.add(key)
                    pool.append(row)

            first_token = probe.split()[0] if probe.split() else ""
            for row in by_first_token.get(first_token, []):
                key = row[1]
                if key not in seen:
                    seen.add(key)
                    pool.append(row)

        if not pool:
            pool = canonicals[:]

        if len(pool) > 350:
            pool = pool[:350]
        return pool

    for item in deduped:
        alias_loose = item.get("alias_loose") or ""
        target_loose = item.get("target_loose") or ""
        scored: list[tuple[float, str]] = []

        for canonical_raw, canonical_loose in _candidate_pool(alias_loose, target_loose):
            score = _score_candidate(alias_loose, target_loose, canonical_loose)
            if score >= min_score:
                scored.append((score, canonical_raw))

        scored.sort(key=lambda entry: entry[0], reverse=True)
        deduped_top: list[tuple[float, str]] = []
        seen_loose = set()
        for score, canonical in scored:
            key = _normalize_loose(canonical)
            if key in seen_loose:
                continue
            seen_loose.add(key)
            deduped_top.append((score, canonical))
            if len(deduped_top) >= 3:
                break

        top = deduped_top
        if not top:
            continue

        suggestions.append(
            {
                "file": item.get("file"),
                "alias": item.get("alias"),
                "target": item.get("target"),
                "suggestions": [
                    {"canonical": canonical, "score": round(score, 4)}
                    for score, canonical in top
                ],
                "best_score": round(top[0][0], 4),
            }
        )

    suggestions.sort(key=lambda row: row.get("best_score", 0), reverse=True)
    top_suggestions = suggestions[:limit]

    return {
        "vocab_root": str(vocab_root),
        "limit": limit,
        "min_score": min_score,
        "unresolved_entity_alias_total": len(deduped),
        "suggestion_count": len(top_suggestions),
        "suggestions": top_suggestions,
    }
