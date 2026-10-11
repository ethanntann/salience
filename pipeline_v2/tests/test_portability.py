"""Guards against the two ways this package has become unimportable off Windows.

CI runs on Linux. The frozen studies once reached ``capture.windows`` through
``capture.collection``, which imported ``ctypes.wintypes`` at module scope, so
merely importing ``pipeline_v2.evaluation`` raised on a non-Windows runner.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]

# Everything reachable from a non-capture entry point. The capture subsystem may
# depend on these; none of them may depend on it.
PIPELINE_MODULES = sorted(
    path.name
    for path in PACKAGE.glob("*.py")
    if path.name not in {"__init__.py", "__main__.py"}
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_no_pipeline_module_imports_the_capture_subsystem():
    offenders = {
        name: sorted(
            module
            for module in _imported_modules(PACKAGE / name)
            if module.startswith("pipeline_v2.capture")
        )
        for name in PIPELINE_MODULES
    }
    assert not {k: v for k, v in offenders.items() if v}


def test_capture_windows_defers_wintypes_to_a_platform_guard():
    source = (PACKAGE / "capture" / "windows.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "ctypes"
        for alias in node.names
    }
    assert "wintypes" not in top_level


def test_display_record_is_constructible_without_windows_apis():
    from pipeline_v2.capture.windows import Display

    display = Display(0, 0, r"\\.\DISPLAY1", 0, 0, 1920, 1080)

    assert (display.right, display.bottom) == (1920, 1080)
