from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BALLOT_TEMPLATE = REPO_ROOT / "webapp" / "templates" / "ballot_lens_f2.html"


def test_ballot_lens_template_has_no_actual_inline_style_attributes():
    source = BALLOT_TEMPLATE.read_text(encoding="utf-8")
    assert re.search(
        r"(?<![-\w])style\s*=",
        source,
        flags=re.IGNORECASE,
    ) is None


def test_ballot_lens_f2_bootstrap_is_csp_safe_and_static():
    source = BALLOT_TEMPLATE.read_text(encoding="utf-8")
    assert 'id="ballotLensF2Root"' in source
    assert 'type="module"' in source
    assert 'data-public-registry-api="/api/public/ballot-lens/registry"' in source
    assert 'data-data-api-url="{{ data_api_url|e }}"' in source

