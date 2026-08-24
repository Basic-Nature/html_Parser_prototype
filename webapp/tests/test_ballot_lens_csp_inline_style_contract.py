from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BALLOT_TEMPLATE = REPO_ROOT / "webapp" / "templates" / "ballot_lens.html"


def test_ballot_lens_template_has_no_actual_inline_style_attributes():
    source = BALLOT_TEMPLATE.read_text(encoding="utf-8")
    assert re.search(
        r"(?<![-\w])style\s*=",
        source,
        flags=re.IGNORECASE,
    ) is None


def test_prompt_status_help_uses_csp_safe_class():
    source = BALLOT_TEMPLATE.read_text(encoding="utf-8")
    assert (
        '<span id="promptStatusChipHelp" class="visually-hidden">'
        in source
    )
