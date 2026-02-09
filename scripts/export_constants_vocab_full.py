from __future__ import annotations

from pathlib import Path

from webapp.parser.Context_Integration.Context_Library import constants


def write_vocab_file(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(values) + "\n"
    path.write_text(text, encoding="utf-8")


def normalize_set(values: set[str]) -> list[str]:
    return sorted({v for v in values if isinstance(v, str) and v.strip()})


def normalize_list(values: list[str]) -> list[str]:
    return [v for v in values if isinstance(v, str) and v.strip()]


def export_vocab_full() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities"

    write_vocab_file(base_dir / "percent_keywords_full.txt", normalize_set(constants.PERCENT_KEYWORDS))
    write_vocab_file(base_dir / "total_keywords_full.txt", normalize_set(constants.TOTAL_KEYWORDS))
    write_vocab_file(base_dir / "misc_footer_keywords_full.txt", normalize_set(constants.MISC_FOOTER_KEYWORDS))

    write_vocab_file(base_dir / "candidate_keywords_full.txt", normalize_set(constants.CANDIDATE_KEYWORDS))
    write_vocab_file(base_dir / "party_keywords_full.txt", normalize_set(constants.PARTY_KEYWORDS))
    write_vocab_file(base_dir / "election_types_full.txt", normalize_set(constants.ELECTION_TYPES))
    write_vocab_file(base_dir / "contest_keywords_full.txt", normalize_set(constants.CONTEST_KEYWORDS))

    write_vocab_file(base_dir / "ballot_types_full.txt", normalize_list(constants.BALLOT_TYPES))
    write_vocab_file(base_dir / "ballot_types_sort_order_full.txt", normalize_list(constants.BALLOT_TYPES_SORT_ORDER))

    write_vocab_file(base_dir / "location_keywords_full.txt", normalize_set(constants.LOCATION_KEYWORDS))
    write_vocab_file(base_dir / "status_keywords_full.txt", normalize_list(constants.STATUS_KEYWORDS))
    write_vocab_file(base_dir / "ballot_measure_types_full.txt", normalize_list(constants.BALLOT_MEASURE_TYPES))
    write_vocab_file(base_dir / "jurisdiction_keywords_full.txt", normalize_list(constants.JURISDICTION_KEYWORDS))
    write_vocab_file(base_dir / "results_keywords_full.txt", normalize_list(constants.RESULTS_KEYWORDS))
    write_vocab_file(base_dir / "election_official_keywords_full.txt", normalize_list(constants.ELECTION_OFFICIAL_KEYWORDS))

    print("Exported full vocab files to", base_dir)


if __name__ == "__main__":
    export_vocab_full()
