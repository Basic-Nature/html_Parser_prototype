"""Small test harness for `apply_patch_safe`.

Run this locally to verify basic behavior of the helper on your workspace.
"""
import json
from pathlib import Path

from tools.agent_patch_utils import apply_patch_safe


def run_tests():
    demo = Path("tools/test_demo_file.txt")
    # Ensure clean start
    try:
        if demo.exists():
            demo.unlink()
    except Exception:
        pass

    print("== Test 1: initial write ==")
    r1 = apply_patch_safe({str(demo): "line1\n"}, auto_apply=True)
    print(json.dumps(r1, indent=2))

    print("== Test 2: no-op write ==")
    r2 = apply_patch_safe({str(demo): "line1\n"}, auto_apply=True)
    print(json.dumps(r2, indent=2))

    print("== Test 3: changed write ==")
    r3 = apply_patch_safe({str(demo): "line1\nline2\n"}, auto_apply=True)
    print(json.dumps(r3, indent=2))

    print("== Test 4: dry-run (auto_apply=False) with change ==")
    r4 = apply_patch_safe({str(demo): "line1\nline2\nline3\n"}, auto_apply=False)
    print(json.dumps(r4, indent=2))

    print("== Done ==")


if __name__ == '__main__':
    run_tests()
