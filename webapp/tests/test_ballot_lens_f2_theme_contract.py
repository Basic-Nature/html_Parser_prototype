from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "webapp" / "frontend" / "ballot-lens"


def read(rel: str) -> str:
    return (FRONTEND / rel).read_text(encoding="utf-8-sig")


def strip_ts_comments(text: str) -> str:
    without_blocks = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\r\n]*", "", without_blocks)


def test_f2t1_cosmic_backdrop_is_mounted_and_styled() -> None:
    main = read("main.tsx")
    app = read("app/AppShell.tsx")
    assert "import './styles/cosmic.css';" in main
    assert "import { CosmicBackdrop }" in app
    assert "<CosmicBackdrop />" in app


def test_f2t1_backdrop_is_decorative_and_noninteractive() -> None:
    component = read("components/theme/CosmicBackdrop.tsx")
    executable = strip_ts_comments(component)

    assert 'aria-hidden="true"' in executable
    assert 'className="blf2-cosmic-backdrop"' in executable

    # The T1 backdrop needs no imports at all. This is stronger and less brittle
    # than banning prose substrings such as the word "socket" in comments.
    assert "import " not in executable

    forbidden_runtime = (
        "useEffect(",
        "useLayoutEffect(",
        "useRef(",
        "useState(",
        "onClick=",
        "onPointer",
        "onMouse",
        "onKey",
        "tabIndex=",
        "contentEditable",
        "<canvas",
        "<button",
        "<a ",
        "<input",
        "<select",
        "<textarea",
        "<form",
        "fetch(",
        "socketClient",
        "socketAdapter",
        "socket.",
    )
    assert all(token not in executable for token in forbidden_runtime)


def test_f2t1_cosmic_css_cannot_intercept_pointer_input() -> None:
    css = read("styles/cosmic.css")
    assert ".blf2-cosmic-backdrop" in css
    assert "pointer-events: none;" in css
    assert "pointer-events: auto" not in css
    assert "cursor: pointer" not in css
    assert ":hover" not in css


def test_f2t1_reduced_motion_and_subtle_light_mode_are_explicit() -> None:
    css = read("styles/cosmic.css")
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "@media (prefers-color-scheme: light)" in css
    assert ".blf2-cosmic-system { opacity: 0.31; }" in css
    assert ".blf2-cosmic-sun::after { opacity: 0.22; }" in css


def test_f2t1_moon_shadow_is_synchronized_to_sun_axis() -> None:
    css = read("styles/cosmic.css")
    assert "Moon shadow synchronization" in css
    assert "blf2-cosmic-moon-orbit 14s linear infinite" in css
    assert "blf2-cosmic-moon-counter-orbit 14s linear infinite" in css
    assert "rotate(360deg)" in css
    assert "rotate(-360deg)" in css
    assert ".blf2-cosmic-moon::after" in css


def test_f2t1_restores_fractal_horizon_and_solar_system_identity() -> None:
    component = read("components/theme/CosmicBackdrop.tsx")
    css = read("styles/cosmic.css")
    for marker in (
        "blf2-cosmic-fractal",
        "blf2-cosmic-horizon",
        "blf2-cosmic-sun",
        "blf2-cosmic-earth",
        "blf2-cosmic-moon",
    ):
        assert marker in component
        assert marker in css
    assert "@keyframes blf2-cosmic-breathe" in css


def test_f2t1_uses_no_external_visual_asset_urls() -> None:
    css = read("styles/cosmic.css")
    component = read("components/theme/CosmicBackdrop.tsx")
    assert "url(" not in css
    assert "http://" not in css
    assert "https://" not in css
    assert "http://" not in component
    assert "https://" not in component


def test_f2t1_cosmic_tokens_preserve_status_color_namespace() -> None:
    tokens = read("styles/tokens.css")
    assert "--blf2-signal-trace:" in tokens
    assert "--blf2-cosmic-cyan:" in tokens
    assert "--blf2-cosmic-sun:" in tokens
    assert "--blf2-success:" in tokens
    assert "--blf2-warning:" in tokens
    assert "--blf2-danger:" in tokens
    cosmic = tokens.split(
        "/* F2-T1 cosmic-observatory visual tokens.", maxsplit=1,
    )[1]
    assert "--blf2-success:" not in cosmic
    assert "--blf2-warning:" not in cosmic
    assert "--blf2-danger:" not in cosmic
