import subprocess


def detect_amd_gpu() -> tuple[bool, str | None]:
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, None
    if completed.returncode != 0:
        return False, None
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    amd = next(
        (name for name in names if "amd" in name.lower() or "radeon" in name.lower()),
        None,
    )
    return amd is not None, amd
