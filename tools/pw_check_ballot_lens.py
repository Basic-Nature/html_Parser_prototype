import os

from playwright.sync_api import sync_playwright

URL = os.environ.get("WEBAPP_URL", "http://127.0.0.1:5000/ballot_lens")
CRITICAL_IDS = [
    "btnNavMore",  # main nav overflow (may be hidden when links fit)
    "parserToolsDropdown",  # parser tools menu
    "btnRunParser2",  # primary run button (modern)
    "btnCancel",  # cancel button presence
]

HIDDEN_ALLOWED = {"btnNavMore", "parserToolsDropdown"}


def visible_state(page, selector: str) -> dict:
    """Return visibility and layout info for a selector."""
    return page.eval_on_selector(
        selector,
        "el => {\n"
        "  const cs = getComputedStyle(el);\n"
        "  const r = el.getBoundingClientRect();\n"
        "  return {\n"
        "    display: cs.display,\n"
        "    visibility: cs.visibility,\n"
        "    opacity: parseFloat(cs.opacity || '0'),\n"
        "    width: r.width,\n"
        "    height: r.height,\n"
        "    x: r.x,\n"
        "    y: r.y\n"
        "  };\n"
        "}",
    )


def assert_present(page, selector: str, *, allow_hidden: bool = False) -> None:
    state = "attached" if allow_hidden else "visible"
    page.wait_for_selector(selector, timeout=10000, state=state)
    if allow_hidden:
        return
    info = visible_state(page, selector)
    if info["display"] == "none":
        raise AssertionError(f"{selector} display none: {info}")
    if info["visibility"] != "visible":
        raise AssertionError(f"{selector} visibility {info['visibility']}: {info}")
    if info["opacity"] <= 0:
        raise AssertionError(f"{selector} opacity {info['opacity']}: {info}")
    if info["width"] <= 0 or info["height"] <= 0:
        raise AssertionError(f"{selector} has zero size: {info}")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Use a lighter wait condition to avoid stalls from long-polling/socket endpoints.
        page.goto(URL, wait_until="load", timeout=45000)
        page.wait_for_selector("#btnNavMore", timeout=15000, state="attached")

        # Basic sanity: page rendered some content
        html_len = len(page.content())
        if html_len < 1000:
            raise AssertionError(f"Page content too small: {html_len}")

        for cid in CRITICAL_IDS:
            selector = f"#{cid}"
            assert_present(page, selector, allow_hidden=(cid in HIDDEN_ALLOWED))

        browser.close()


if __name__ == "__main__":
    main()
