from __future__ import annotations

from pathlib import Path

from webapp.parser.Context_Integration.Context_Library import constants


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "webapp" / "parser" / "Context_Integration" / "vocab" / "entities"

    state_to_county = constants.KNOWN_STATE_TO_COUNTY_MAP
    county_to_precinct = constants.KNOWN_COUNTY_TO_PRECINCTS_MAP

    states = sorted(state_to_county.keys())
    state_lines = [s for s in states if s]

    county_lines: list[str] = []
    for state in states:
        counties = state_to_county.get(state, [])
        for county in counties:
            county_lines.append(f"{state}|{county}")

    county_precinct_lines: list[str] = []
    for county, precincts in county_to_precinct.items():
        for precinct in precincts:
            county_precinct_lines.append(f"{county}|{precinct}")

    _write_lines(base_dir / "states.txt", state_lines)
    _write_lines(base_dir / "counties_by_state.txt", county_lines)
    _write_lines(base_dir / "county_precincts.txt", county_precinct_lines)

    print(f"Wrote {len(state_lines)} states")
    print(f"Wrote {len(county_lines)} state-county rows")
    print(f"Wrote {len(county_precinct_lines)} county-precinct rows")


if __name__ == "__main__":
    main()
