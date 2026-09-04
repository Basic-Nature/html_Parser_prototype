from pathlib import Path
ROOT=Path(".")
WEBAPP=ROOT/"webapp/templates/quick_reference.html"; DOCS=ROOT/"docs/quick_reference.html"; CSS=ROOT/"webapp/static/css/quick_reference.css"; PANEL=ROOT/"webapp/frontend/ballot-lens/components/source/SourcePanel.tsx"; CHECKPOINTS=ROOT/"webapp/parser/services/ballot_lens_checkpoint_runtime.py"
STALE=("Direct URLs","Batch Mode","Output Bypass","Clone Session","Filter Presets","URL drafts","Maximum 20 URLs","v3.0","badge-new","NEW</span>")
MODES=(("Registry","Approved public sources","Public"),("Upload","Local election artifact","Trusted"),("URL Library","Approved trusted targets","Trusted"),("Worklist","Governed queue handoff","Trusted"))
CP=("Resolve Source","Provider Detection","Acquire","Detect Structure","Contest Selection","Vote Method Selection","Normalize","Validate","Preview")
def r(p): return p.read_text(encoding="utf-8-sig")
def test_identity_and_modes_match_f2():
    panel=r(PANEL)
    for p in (WEBAPP,DOCS):
        s=r(p); assert "ElectionPulse — Ballot Lens Quick Reference" in s; assert "F2 primary route" in s
        for label,desc,auth in MODES: assert label in panel and desc in panel and f">{label}<" in s and desc in s and f">{auth}<" in s
def test_checkpoints_match_runtime():
    runtime=r(CHECKPOINTS)
    for label in CP:
        assert f'"{label}"' in runtime
        assert label in r(WEBAPP) and label in r(DOCS)
def test_session_and_validation_authority():
    for p in (WEBAPP,DOCS):
        s=r(p); assert "Run approved source" in s; assert "must exactly match" in s; assert "The server creates the active session" in s; assert "Foreign or stale events" in s; assert "Session History" in s and "view-only" in s; assert "does not resume it, rebind parser authority" in s; assert "EXACT_MATCH" in s and "UNRESOLVED" in s; assert "not automatically a mismatch" in s; assert "distinct from numeric zero" in s
def test_legacy_guidance_removed():
    for p in (WEBAPP,DOCS):
        s=r(p)
        for x in STALE: assert x not in s
def test_external_assets_and_feature_finder_first_table():
    w=r(WEBAPP); d=r(DOCS)
    assert "style=" not in w and "style=" not in d and "<script>" not in w and "<script>" not in d
    assert "css/quick_reference.css" in w and "js/quick_reference.js" in w
    assert "../webapp/static/css/quick_reference.css" in d and "../webapp/static/js/quick_reference.js" in d
    for s in (w,d): assert s.index('id="feature-finder"') < s.index("<table>") < s.index('id="keyboard-shortcuts"')
def test_css_supports_f2_layout():
    s=r(CSS)
    for x in (".hero",".status-pill",".flow",".authority-grid",".source-badge.public",".source-badge.trusted",".checkpoint-grid",".callout.warning",".callout.success"): assert x in s
