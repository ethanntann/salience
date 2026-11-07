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

import hashlib
import io
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "Salience"
DIST = ROOT / "dist"
# One version for the app and its installer: `pyproject.toml`'s.
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
# What the app imports at run time (`pyproject.toml` extras audio, app, capture),
# at the exact versions it was tested with, dependencies included. Not
# windows-capture: the Python recorder for an FFmpeg older than 8, never used
# with the FFmpeg bundled here, and it brings 115 MB of OpenCV.
LOCK = ROOT / "installer" / "requirements.lock"
# The embeddable Python's SHA-256, so a tampered download fails the build.
# python.org lists only an MD5 (85a5dcd0350f7ca4e2876295b0b281e3), checked
# against this download before its SHA-256 was taken (2026-11-07).
PYTHON_SHA256 = {
    "3.14.2": "f05e28d161c6b15af64a7cb7f08b4a22b3a6b03eee71baee24ea557b3bdd5798",
}
APP_PACKAGES = ["pipeline_v2", "salience_media"]


def python_embed(target: Path) -> None:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if version not in PYTHON_SHA256:
        raise SystemExit(f"no checksum for Python {version}: add one to PYTHON_SHA256")
    url = f"https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip"
    print(f"python {version}: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != PYTHON_SHA256[version]:
        raise SystemExit(f"{url} does not match its checksum: not building with it")
    zipfile.ZipFile(io.BytesIO(data)).extractall(target)
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
    # Exact versions with no dependency resolving: `--only-binary :all:` once
    # made pip skip pywebview 6 (its proxy_tools has no wheel) for 3.4, from
    # 2020, whose window could not open (2026-11-07).
    site = target / "Lib" / "site-packages"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location",
                    "--no-deps", "--target", str(site), "-r", str(LOCK)], check=True)
    # Libraries' own tests: not needed to run, large, and ONNX's test data has
    # names long enough to pass Windows' 260-character path limit, which made
    # the installer roll back (2026-11-06).
    for junk in [*site.glob("**/tests"), site / "onnx" / "backend" / "test"]:
        if junk.is_dir():
            shutil.rmtree(junk, ignore_errors=True)
    # What was installed is what the lock says, or the build stops.
    found = subprocess.run([sys.executable, "-m", "pip", "list", "--path", str(site), "--format=freeze"],
                           capture_output=True, text=True, check=True).stdout
    def pins(text: str) -> set[str]:
        return {line.strip().lower().replace("_", "-") for line in text.splitlines()
                if line.strip() and not line.startswith("#")}
    if pins(found) != pins(LOCK.read_text(encoding="utf-8")):
        raise SystemExit(f"installed packages differ from {LOCK.name}: {sorted(pins(found) ^ pins(LOCK.read_text(encoding='utf-8')))}")
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
    digest = hashlib.sha256(setup.read_bytes()).hexdigest()
    # Published beside the release, so a tester can check their download.
    setup.with_name(setup.name + ".sha256").write_text(f"{digest}  {setup.name}\n", encoding="utf-8")
    print(f"installer: {setup} ({setup.stat().st_size / 1e6:.0f} MB), sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
