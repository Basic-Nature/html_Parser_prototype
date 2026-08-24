from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
READER = REPO_ROOT / "webapp" / "parser" / "services" / "canonical_election_reader.py"
BLUEPRINT = REPO_ROOT / "webapp" / "parser" / "routes" / "data_framework_blueprint.py"
APP = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_canonical_facets_are_read_only_and_self_excluding():
    reader = _read(READER)

    assert "class CanonicalFacetFilters:" in reader
    assert "def query_canonical_facets(" in reader
    assert 'exclude="year"' in reader
    assert 'exclude="state"' in reader
    assert 'exclude="jurisdiction"' in reader
    assert 'exclude="contest"' in reader
    assert 'conn.exec_driver_sql("SET TRANSACTION READ ONLY")' in reader
    assert "# Facet discovery is read-only and never commits." in reader
    assert "transaction.rollback()" in reader


def test_canonical_facets_preserve_jurisdiction_name_and_type():
    reader = _read(READER)

    assert 'result.jurisdiction_name.label("jurisdiction_name")' in reader
    assert 'result.jurisdiction_type.label("jurisdiction_type")' in reader
    assert '"name": str(row.jurisdiction_name)' in reader
    assert '"type": row.jurisdiction_type' in reader
    assert '"county"' not in reader.split("def query_canonical_facets(", 1)[1].split("def query_canonical_results(", 1)[0]


def test_data_framework_blueprint_exposes_canonical_facets_get_only():
    blueprint = _read(BLUEPRINT)

    assert '@bp.route("/api/data_framework/canonical_facets", methods=["GET"], endpoint="api_data_framework_canonical_facets")' in blueprint
    assert 'return _call_handler("api_data_framework_canonical_facets")' in blueprint


def test_application_registers_canonical_facets_handler_without_lineage_claim():
    app = _read(APP)

    assert "def api_data_framework_canonical_facets():" in app
    assert '"api_data_framework_canonical_facets": api_data_framework_canonical_facets,' in app
    assert '"contract": "canonical_facets_v1"' in app
    assert '"authority": "canonical_production"' in app
    assert '"facet_mode": "self_excluding"' in app
    assert '"lineage": "not_inferred"' in app
    assert "query_canonical_facets(" in app
