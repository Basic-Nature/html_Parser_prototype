from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from webapp.parser.contracts.artifact_identity import (
    ARTIFACT_IDENTITY_ALGORITHM,
    ARTIFACT_IDENTITY_CANONICAL_HEX_CASE,
    ARTIFACT_IDENTITY_HANDOFF_CONTRACT,
    ARTIFACT_IDENTITY_SEMANTICS,
    INVALID_ARTIFACT_IDENTITY_POLICY,
    MISSING_ARTIFACT_IDENTITY_POLICY,
    ArtifactIdentityHandoff,
)


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "parser"
    / "contracts"
    / "artifact_identity.py"
)


def test_contract_constants_are_exact():
    assert (
        ARTIFACT_IDENTITY_HANDOFF_CONTRACT
        == "artifact_identity_handoff_v1"
    )
    assert ARTIFACT_IDENTITY_ALGORITHM == "sha256"
    assert (
        ARTIFACT_IDENTITY_SEMANTICS
        == "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
    )
    assert ARTIFACT_IDENTITY_CANONICAL_HEX_CASE == "lowercase"
    assert (
        MISSING_ARTIFACT_IDENTITY_POLICY
        == "PRESERVE_UNKNOWN_DO_NOT_HASH_FOR_INSTRUMENTATION"
    )
    assert (
        INVALID_ARTIFACT_IDENTITY_POLICY
        == "REJECT_INVALID_FORMAT_NO_INFERENCE"
    )


def test_valid_lowercase_identity_is_preserved():
    value = "a" * 64
    handoff = ArtifactIdentityHandoff(value)

    assert handoff.document_sha256 == value
    assert handoff.algorithm == "sha256"
    assert (
        handoff.semantics
        == "SHA256_OF_IMMUTABLE_CONTENT_BYTES"
    )


def test_uppercase_hex_is_canonicalized_to_lowercase():
    handoff = ArtifactIdentityHandoff("A" * 64)

    assert handoff.document_sha256 == "a" * 64


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "a" * 63,
        "a" * 65,
        "g" * 64,
        123,
    ],
)
def test_invalid_document_sha256_is_rejected(value):
    with pytest.raises(ValueError):
        ArtifactIdentityHandoff(value)  # type: ignore[arg-type]


def test_algorithm_override_is_rejected():
    with pytest.raises(ValueError):
        ArtifactIdentityHandoff(
            "b" * 64,
            algorithm="sha512",
        )


def test_semantics_override_is_rejected():
    with pytest.raises(ValueError):
        ArtifactIdentityHandoff(
            "b" * 64,
            semantics="PATH_HASH",
        )


def test_contract_is_frozen():
    handoff = ArtifactIdentityHandoff("c" * 64)

    with pytest.raises(FrozenInstanceError):
        handoff.document_sha256 = "d" * 64  # type: ignore[misc]


def test_contract_fields_are_identity_only():
    names = [field.name for field in fields(ArtifactIdentityHandoff)]

    assert names == [
        "document_sha256",
        "algorithm",
        "semantics",
    ]

    forbidden = {
        "path",
        "pdf_path",
        "url",
        "source_url",
        "locator",
        "filename",
        "file_path",
    }

    assert not (forbidden & set(names))


def test_contract_source_has_no_hashing_io_or_runtime_authority():
    source = CONTRACT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CONTRACT_PATH))

    imported_roots = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_roots.add(module.split(".", 1)[0])

    assert "hashlib" not in imported_roots
    assert "pathlib" not in imported_roots
    assert "requests" not in imported_roots
    assert "fitz" not in imported_roots

    call_names = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            call_names.append(node.func.attr)

    assert "open" not in call_names
    assert "sha256" not in call_names
    assert "_get_page_orientation_map" not in call_names
    assert "_collect_page_orientation" not in call_names
    assert "parse_pdf_election_results" not in call_names

    assert "require_sha256" in call_names
