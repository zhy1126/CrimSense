"""The manuscript text is the single source of behavioral instructions."""

from pathlib import Path
import re

ROOT = Path(__file__).with_name("prompts")
CITIES = ("chicago", "dallas", "los_angeles")
VERSION = "manuscript-2026-10-08"


def text(name):
    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError("Unknown prompt block")
    return (ROOT / (name + ".txt")).read_text().strip()


def render(name, **values):
    source = text(name)
    required = set(re.findall(r"\{\{([A-Z_]+)\}\}", source))
    if required - values.keys():
        raise ValueError(
            "Missing prompt values: " + str(sorted(required - values.keys()))
        )
    # Substitute template placeholders once; user-provided facts are never reparsed as templates.
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: str(values[m[1]]), source)


def city_context(city):
    if city not in CITIES:
        raise ValueError("Unsupported city")
    return text(city + "_context")
