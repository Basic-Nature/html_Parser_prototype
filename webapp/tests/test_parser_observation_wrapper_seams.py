from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from webapp.parser.handlers.formats import csv_handler, xlsx_handler
import webapp.parser.config as parser_config


def _callback():
    return lambda payload: None


def _patch_provided_tables_dependencies(monkeypatch, module, *, normalize: bool):
    monkeypatch.setattr(
        module,
        "robust_table_extraction",
        lambda page=None, extraction_context=None: (["Candidate"], [{"Candidate": "A"}]),
    )
    if normalize:
        monkeypatch.setattr(
            module,
            "normalize_table_headers",
            lambda headers, rows: (list(headers), [dict(row) for row in rows]),
        )
    monkeypatch.setattr(
        module,
        "build_table_noninteractive",
        lambda **kwargs: (
            ["Candidate"],
            [{"Candidate": "A"}],
            {"entity_info": {"candidate_count": 1}},
        ),
    )
    monkeypatch.setattr(
        module,
        "finalize_election_output",
        lambda **kwargs: {
            "csv_path": "output/projected.csv",
            "metadata_path": "output/projected.metadata.json",
        },
    )
    monkeypatch.setattr(
        parser_config,
        "log_extraction_quality",
        lambda *args, **kwargs: {"projected": True},
    )


def test_csv_wrapper_named_callback_parameter_remains_absent():
    assert "parser_observation_emit_func" not in inspect.signature(
        csv_handler.parse
    ).parameters
    assert "parser_observation_emit_func" in inspect.signature(
        csv_handler.parse_csv_election_results
    ).parameters


def test_xlsx_wrapper_named_callback_parameter_remains_absent():
    assert "parser_observation_emit_func" not in inspect.signature(
        xlsx_handler.parse
    ).parameters
    assert "parser_observation_emit_func" in inspect.signature(
        xlsx_handler.parse_xlsx_election_results
    ).parameters


def test_csv_manual_file_extracts_generic_kwarg_and_forwards_once(monkeypatch, tmp_path):
    source = tmp_path / "fixture.csv"
    source.write_text("Candidate\nA\n", encoding="utf-8")
    captured = []

    def fake_primary(*args, **kwargs):
        captured.append(kwargs.get("parser_observation_emit_func"))
        return ["Candidate"], [{"Candidate": "A"}], "Contest", {"handler": "csv_handler"}

    monkeypatch.setattr(csv_handler, "parse_csv_election_results", fake_primary)
    callback = _callback()

    result = csv_handler.parse(
        manual_file=str(source),
        html_context={},
        parser_observation_emit_func=callback,
    )

    assert result[2] == "Contest"
    assert captured == [callback]


def test_xlsx_manual_file_extracts_generic_kwarg_and_forwards_once(monkeypatch, tmp_path):
    source = tmp_path / "fixture.xlsx"
    source.write_bytes(b"projection-fixture")
    captured = []

    def fake_primary(*args, **kwargs):
        captured.append(kwargs.get("parser_observation_emit_func"))
        return ["Candidate"], [{"Candidate": "A"}], "Contest", {"handler": "xlsx_handler"}

    monkeypatch.setattr(xlsx_handler, "parse_xlsx_election_results", fake_primary)
    callback = _callback()

    result = xlsx_handler.parse(
        manual_file=str(source),
        html_context={},
        parser_observation_emit_func=callback,
    )

    assert result[2] == "Contest"
    assert captured == [callback]


def test_csv_provided_tables_with_callback_fails_closed():
    with pytest.raises(RuntimeError, match="provided_tables wrapper path"):
        csv_handler.parse(
            html_context={"provided_tables": [{"Candidate": "A"}]},
            parser_observation_emit_func=_callback(),
        )


def test_xlsx_provided_tables_with_callback_fails_closed():
    with pytest.raises(RuntimeError, match="provided_tables wrapper path"):
        xlsx_handler.parse(
            html_context={"provided_tables": [{"Candidate": "A"}]},
            parser_observation_emit_func=_callback(),
        )


def test_csv_provided_tables_without_callback_preserves_current_path(monkeypatch):
    _patch_provided_tables_dependencies(monkeypatch, csv_handler, normalize=True)

    headers, rows, contest, metadata = csv_handler.parse(
        html_context={
            "provided_tables": [{"Candidate": "A"}],
            "contest": "Contest",
            "state": "State",
            "county": "County",
        },
    )

    assert headers == ["Candidate"]
    assert rows == [{"Candidate": "A"}]
    assert contest == "Contest"
    assert metadata["handler"] == "csv_handler"
    assert metadata["quality_metrics"] == {"projected": True}


def test_xlsx_provided_tables_without_callback_preserves_current_path(monkeypatch):
    _patch_provided_tables_dependencies(monkeypatch, xlsx_handler, normalize=False)

    headers, rows, contest, metadata = xlsx_handler.parse(
        html_context={
            "provided_tables": [{"Candidate": "A"}],
            "contest": "Contest",
            "state": "State",
            "county": "County",
        },
    )

    assert headers == ["Candidate"]
    assert rows == [{"Candidate": "A"}]
    assert contest == "Contest"
    assert metadata["handler"] == "xlsx_handler"
    assert metadata["quality_metrics"] == {"projected": True}


def test_projection_does_not_add_router_runtime_or_socket_callback_references():
    webapp_root = Path(__file__).resolve().parents[1]
    for rel in (
        "parser/utils/format_router.py",
        "parser/state_router.py",
        "parser/utils/shared_logic.py",
        "parser/html_election_parser.py",
        "parser/web_pipeline.py",
        "parser/socket_ballot_lens_orchestration.py",
    ):
        text = (webapp_root / rel).read_text(encoding="utf-8")
        assert "parser_observation_emit_func" not in text
