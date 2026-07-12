from pathlib import Path
import json


def load_demo_clips(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Demo clip seed must be a list")
    return data
