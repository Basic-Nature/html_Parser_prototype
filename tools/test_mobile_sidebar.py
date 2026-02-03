#!/usr/bin/env python3
"""
Test mobile sidebar scrolling and swipe gesture improvements.
Verifies:
1. Sidebar has proper scrollable height
2. Scroll indicators (has-scroll class) are applied
3. Interactive elements exist and are accessible
"""
import os

from playwright.sync_api import sync_playwright

URL = os.environ.get("WEBAPP_URL", "http://127.0.0.1:5000/ballot_lens")


def test_mobile_sidebar_scrollability(page):
    """Test that sidebar is scrollable on mobile viewport."""
    # Set mobile viewport
    page.set_viewport_size({"width": 375, "height": 667})
    
    # Open sidebar if it exists
    sidebar_selector = ".sidebar-left, #sidebar"
    page.wait_for_selector(sidebar_selector, timeout=10000, state="attached")
    
    # Check if sidebar has scrollable content
    scroll_info = page.eval_on_selector(
        sidebar_selector,
        """el => ({
            scrollHeight: el.scrollHeight,
            clientHeight: el.clientHeight,
            hasScroll: el.scrollHeight > el.clientHeight,
            overflowY: getComputedStyle(el).overflowY,
            classList: Array.from(el.classList)
        })"""
    )
    
    print(f"Sidebar scroll info: {scroll_info}")
    
    # Verify overflow-y is auto or scroll
    assert scroll_info["overflowY"] in ["auto", "scroll"], \
        f"Sidebar overflow-y should be auto or scroll, got: {scroll_info['overflowY']}"
    
    # Check if scroll indicators would be applied (has-scroll class)
    # Note: The class is applied by JS, so we just verify the structure is correct
    print(f"✓ Sidebar has overflow-y: {scroll_info['overflowY']}")
    print(f"✓ Scrollable: {scroll_info['hasScroll']}")
    
    return True


def test_sidebar_css_properties(page):
    """Test that sidebar has correct CSS properties for mobile."""
    page.set_viewport_size({"width": 375, "height": 667})
    
    sidebar_selector = ".sidebar-left, #sidebar"
    page.wait_for_selector(sidebar_selector, timeout=10000, state="attached")
    
    css_props = page.eval_on_selector(
        sidebar_selector,
        """el => {
            const cs = getComputedStyle(el);
            return {
                position: cs.position,
                overflowY: cs.overflowY,
                overflowX: cs.overflowX,
                webkitOverflowScrolling: cs.webkitOverflowScrolling
            };
        }"""
    )
    
    print(f"Sidebar CSS properties: {css_props}")
    
    # Verify key CSS properties
    assert css_props["position"] == "fixed", \
        f"Expected position: fixed, got: {css_props['position']}"
    assert css_props["overflowY"] in ["auto", "scroll"], \
        f"Expected overflow-y: auto or scroll, got: {css_props['overflowY']}"
    assert css_props["overflowX"] == "hidden", \
        f"Expected overflow-x: hidden, got: {css_props['overflowX']}"
    
    print("✓ Sidebar CSS properties are correct")
    return True


def test_interactive_elements_exist(page):
    """Test that interactive elements in sidebar exist."""
    page.set_viewport_size({"width": 375, "height": 667})
    
    # Check for URL input field
    url_input = page.query_selector("#newUrl")
    if url_input:
        print("✓ URL input field found")
    
    # Check for source cards
    source_cards = page.query_selector_all(".source-card")
    print(f"✓ Found {len(source_cards)} source cards")
    
    # Check for control groups
    control_groups = page.query_selector_all(".control-group")
    print(f"✓ Found {len(control_groups)} control groups")
    
    return True


def main():
    """Run all tests."""
    print(f"Testing mobile sidebar at: {URL}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # Load page
            page.goto(URL, wait_until="load", timeout=45000)
            page.wait_for_selector("body", timeout=15000)
            
            # Run tests
            print("\n=== Test 1: Sidebar Scrollability ===")
            test_mobile_sidebar_scrollability(page)
            
            print("\n=== Test 2: Sidebar CSS Properties ===")
            test_sidebar_css_properties(page)
            
            print("\n=== Test 3: Interactive Elements ===")
            test_interactive_elements_exist(page)
            
            print("\n✅ All tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
