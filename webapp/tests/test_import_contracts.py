"""Permanent import smoke contracts recovered from tests/test_imports.py."""

from __future__ import annotations


def test_status_reconciliation_import_contract():
    from webapp.parser.utils.status_reconciliation import StatusReconciliation

    assert StatusReconciliation is not None


def test_webapp_app_import_contract():
    from webapp.Smart_Elections_Parser_Webapp import app

    assert app is not None