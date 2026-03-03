from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ComparisonDifference:
    candidate: str
    field: str
    dl1_value: Any
    dl2_value: Any
    abs_diff: float | None = None
    within_tolerance: bool = False


@dataclass
class ComparisonResult:
    exact_matches: int = 0
    near_matches: int = 0
    mismatches: list[ComparisonDifference] = field(default_factory=list)
    missing_candidates: list[str] = field(default_factory=list)
    extra_candidates: list[str] = field(default_factory=list)
    vote_diff_summary: dict[str, float] = field(default_factory=dict)
    total_dl1_candidates: int = 0
    total_dl2_candidates: int = 0
    accuracy: float = 0.0


class DataComparator:
    """Compare parser output (DL2) against verified ground truth (DL1)."""

    DEFAULT_TOLERANCE: dict[str, float] = {
        "votes": 5.0,
        "percent": 0.2,
    }

    @staticmethod
    def _canonical_name(value: Any) -> str:
        text = str(value or "").strip().lower()
        return " ".join(text.split())

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            cleaned = str(value).replace(",", "").replace("%", "").strip()
            if not cleaned:
                return None
            return float(cleaned)
        except Exception:
            return None

    def _normalize_candidates(self, payload: Any) -> dict[str, dict[str, Any]]:
        if isinstance(payload, dict):
            candidates = payload.get("candidates")
            if isinstance(candidates, list):
                rows = candidates
            else:
                rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        normalized: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate_name = (
                row.get("name")
                or row.get("candidate")
                or row.get("candidate_name")
                or row.get("Candidate")
            )
            key = self._canonical_name(candidate_name)
            if not key:
                continue
            normalized[key] = {
                "name": str(candidate_name).strip(),
                "votes": self._to_float(row.get("votes") if row.get("votes") is not None else row.get("vote_count")),
                "percent": self._to_float(row.get("percent") if row.get("percent") is not None else row.get("percentage")),
                "party": (row.get("party") or row.get("Party") or "").strip() if isinstance((row.get("party") or row.get("Party") or ""), str) else row.get("party") or row.get("Party"),
            }
        return normalized

    def compare_datasets(
        self,
        dl1_data: dict[str, Any] | list[dict[str, Any]],
        dl2_data: dict[str, Any] | list[dict[str, Any]],
        tolerance: dict[str, float] | None = None,
    ) -> ComparisonResult:
        tol = dict(self.DEFAULT_TOLERANCE)
        if isinstance(tolerance, dict):
            tol.update({k: float(v) for k, v in tolerance.items() if v is not None})

        dl1 = self._normalize_candidates(dl1_data)
        dl2 = self._normalize_candidates(dl2_data)

        result = ComparisonResult(
            total_dl1_candidates=len(dl1),
            total_dl2_candidates=len(dl2),
        )

        vote_diffs: list[float] = []

        for key, dl1_row in dl1.items():
            if key not in dl2:
                result.missing_candidates.append(dl1_row["name"])
                continue

            dl2_row = dl2[key]
            candidate_differences: list[ComparisonDifference] = []

            # votes
            dl1_votes = dl1_row.get("votes")
            dl2_votes = dl2_row.get("votes")
            if dl1_votes is not None and dl2_votes is not None:
                diff_votes = abs(float(dl1_votes) - float(dl2_votes))
                vote_diffs.append(diff_votes)
                if diff_votes != 0:
                    candidate_differences.append(
                        ComparisonDifference(
                            candidate=dl1_row["name"],
                            field="votes",
                            dl1_value=dl1_votes,
                            dl2_value=dl2_votes,
                            abs_diff=diff_votes,
                            within_tolerance=diff_votes <= tol["votes"],
                        )
                    )

            # percent
            dl1_percent = dl1_row.get("percent")
            dl2_percent = dl2_row.get("percent")
            if dl1_percent is not None and dl2_percent is not None:
                diff_percent = abs(float(dl1_percent) - float(dl2_percent))
                if diff_percent != 0:
                    candidate_differences.append(
                        ComparisonDifference(
                            candidate=dl1_row["name"],
                            field="percent",
                            dl1_value=dl1_percent,
                            dl2_value=dl2_percent,
                            abs_diff=diff_percent,
                            within_tolerance=diff_percent <= tol["percent"],
                        )
                    )

            # party (strict string match when both present)
            dl1_party = str(dl1_row.get("party") or "").strip().lower()
            dl2_party = str(dl2_row.get("party") or "").strip().lower()
            if dl1_party and dl2_party and dl1_party != dl2_party:
                candidate_differences.append(
                    ComparisonDifference(
                        candidate=dl1_row["name"],
                        field="party",
                        dl1_value=dl1_row.get("party"),
                        dl2_value=dl2_row.get("party"),
                        abs_diff=None,
                        within_tolerance=False,
                    )
                )

            if not candidate_differences:
                result.exact_matches += 1
                continue

            if all(d.within_tolerance for d in candidate_differences if d.abs_diff is not None) and not any(d.field == "party" and not d.within_tolerance for d in candidate_differences):
                result.near_matches += 1
            else:
                result.mismatches.extend(candidate_differences)

        for key, dl2_row in dl2.items():
            if key not in dl1:
                result.extra_candidates.append(dl2_row["name"])

        compared_total = max(result.total_dl1_candidates, 1)
        result.accuracy = round((result.exact_matches + result.near_matches) / compared_total, 6)
        if vote_diffs:
            result.vote_diff_summary = {
                "avg_vote_diff": round(sum(vote_diffs) / len(vote_diffs), 4),
                "max_vote_diff": round(max(vote_diffs), 4),
            }
        else:
            result.vote_diff_summary = {
                "avg_vote_diff": 0.0,
                "max_vote_diff": 0.0,
            }

        return result

    def evaluate_regression(
        self,
        result: ComparisonResult,
        *,
        min_accuracy: float = 0.95,
        max_mismatches: int = 0,
    ) -> dict[str, Any]:
        failures: list[str] = []
        if result.accuracy < min_accuracy:
            failures.append(f"accuracy_below_threshold:{result.accuracy:.4f}<{min_accuracy:.4f}")
        if len(result.mismatches) > max_mismatches:
            failures.append(f"mismatches_above_threshold:{len(result.mismatches)}>{max_mismatches}")
        return {
            "status": "pass" if not failures else "fail",
            "min_accuracy": min_accuracy,
            "max_mismatches": max_mismatches,
            "failure_reasons": failures,
        }

    def build_regression_report(
        self,
        result: ComparisonResult,
        *,
        context: dict[str, Any] | None = None,
        min_accuracy: float = 0.95,
        max_mismatches: int = 0,
    ) -> dict[str, Any]:
        gate = self.evaluate_regression(
            result,
            min_accuracy=min_accuracy,
            max_mismatches=max_mismatches,
        )
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_dl1_candidates": result.total_dl1_candidates,
                "total_dl2_candidates": result.total_dl2_candidates,
                "exact_matches": result.exact_matches,
                "near_matches": result.near_matches,
                "mismatch_count": len(result.mismatches),
                "missing_candidates_count": len(result.missing_candidates),
                "extra_candidates_count": len(result.extra_candidates),
                "accuracy": result.accuracy,
            },
            "gate": gate,
            "vote_diff_summary": result.vote_diff_summary,
            "missing_candidates": result.missing_candidates,
            "extra_candidates": result.extra_candidates,
            "mismatches": [asdict(m) for m in result.mismatches],
            "context": context or {},
        }
