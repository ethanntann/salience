"""Build Salience-Setup.exe: Python, the app and FFmpeg in one installer.

    python installer/build.py

A tester installs nothing else: no Python, no FFmpeg, no command line. The
app goes in the player's own folder (no admin rights), with a Start menu
entry and an uninstaller. Needs Inno Setup 6 (`winget install
JRSoftware.InnoSetup`) and FFmpeg 8+ on PATH, which is copied in.

Python is the official embeddable build, with tkinter (the first-run dialogs)
copied from this PC's Python of the same version, because the embeddable
build leaves it out. Only what the app imports is installed: no PyTorch.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "Salience"
DIST = ROOT / "dist"
VERSION = "0.3.0"
# What the app imports at run time (`pyproject.toml` extras audio, app, capture).
PACKAGES = [
    "numpy", "pydantic", "scikit-learn", "scipy",
    # Not windows-capture: the Python recorder for an FFmpeg older than 8, never
    # used with the FFmpeg bundled here, and it brings 115 MB of OpenCV.
    "soundcard", "soundfile",
    "onnx-asr[hub]", "onnx", "onnxruntime-directml",
    "pystray", "pillow", "pywebview",
]
APP_PACKAGES = ["pipeline_v2", "salience_media"]


def python_embed(target: Path) -> None:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    url = f"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip"
    print(f"python {version}: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        zipfile.ZipFile(io.BytesIO(response.read())).extractall(target)
    # tkinter, which the embeddable build leaves out.
    base = Path(sys.base_prefix)
    shutil.copytree(base / "Lib" / "tkinter", target / "Lib" / "tkinter",
                    ignore=shutil.ignore_patterns("test*", "__pycache__"))
    for name in ("_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "zlib1.dll"):
        if (base / "DLLs" / name).is_file():
            shutil.copy2(base / "DLLs" / name, target / name)
    shutil.copytree(base / "tcl", target / "tcl")
    pth = next(target.glob("python*._pth"))
    stdlib = next(target.glob("python*.zip")).name
    pth.write_text("\n".join([stdlib, ".", "Lib", "Lib\\site-packages", "..\\app", "import site", ""]),
                   encoding="utf-8")


def packages(target: Path) -> None:
    site = target / "Lib" / "site-packages"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location",
                    "--only-binary", ":all:", "--target", str(site), *PACKAGES], check=True)
    # Libraries' own tests: not needed to run, large, and ONNX's test data has
    # names long enough to pass Windows' 260-character path limit, which made
    # the installer roll back (2026-11-06).
    for junk in [*site.glob("**/tests"), site / "onnx" / "backend" / "test"]:
        if junk.is_dir():
            shutil.rmtree(junk, ignore_errors=True)
    longest = max(len(str(f.relative_to(target.parent))) for f in site.rglob("*"))
    print(f"longest path inside the install: {longest} characters")


def app_code(target: Path) -> None:
    for name in APP_PACKAGES:
        shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns(
            "tests", "__pycache__", "*.pyc", ".data", "*.md"))


def ffmpeg(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit("ffmpeg is not on PATH")
    binary = Path(found).resolve().parent
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        shutil.copy2(binary / name, target / name)
    # FFmpeg's licence travels with it; the README says where its source is.
    for name in ("LICENSE", "README.txt"):
        if (binary.parent / name).is_file():
            shutil.copy2(binary.parent / name, target / name)


def icon(path: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from pipeline_v2.capture.app import _icon_image

    _icon_image(False).save(path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])


def inno() -> Path:
    candidates = [Path.home() / "AppData/Local/Programs/Inno Setup 6/ISCC.exe",
                  Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe")]
    compiler = next((c for c in candidates if c.is_file()), None)
    if compiler is None:
        raise SystemExit("Inno Setup 6 is not installed: winget install JRSoftware.InnoSetup")
    DIST.mkdir(exist_ok=True)
    subprocess.run([str(compiler), "/Q", f"/DBuildDir={BUILD}", f"/DOutDir={DIST}", f"/DAppVersion={VERSION}",
                    str(ROOT / "installer" / "salience.iss")], check=True)
    return DIST / f"Salience-Setup-{VERSION}.exe"


def main() -> int:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    python_embed(BUILD / "python")
    packages(BUILD / "python")
    app_code(BUILD / "app")
    ffmpeg(BUILD / "ffmpeg")
    icon(BUILD / "salience.ico")
    size = sum(f.stat().st_size for f in BUILD.rglob("*") if f.is_file())
    print(f"bundle: {size / 1e6:.0f} MB")
    setup = inno()
    print(f"installer: {setup} ({setup.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
