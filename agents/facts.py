"""Validate observed encounter fields without injecting unmeasured traits."""

import copy
import json
import math
from simulation.geography import ENV_FIELDS
from simulation.io import digest as _sha

_PERSON = {
    "agent_id",
    "gender",
    "race",
    "residence",
    "income_level",
    "current_location",
    "historical_trajectory",
    "visited_locations",
}


_HRI = _PERSON | {"criminal_record", "motivation"}


def _text(x):
    if not isinstance(x, str) or not x.strip():
        raise ValueError("Expected nonempty text")


def _person(row, location, step, *, hri=False):
    if not isinstance(row, dict) or set(row) != (_HRI if hri else _PERSON):
        raise ValueError("Unapproved person fields")
    for field in ("agent_id", "gender", "race", "residence", "current_location"):
        _text(row[field])
    if ":" in row["agent_id"] or row["current_location"] != location:
        raise ValueError("Invalid identity or target location")
    if not isinstance(row["income_level"], (str, int)) or isinstance(
        row["income_level"], bool
    ):
        raise ValueError("Invalid source income level")
    trajectory = row["historical_trajectory"]
    if not isinstance(trajectory, (list, tuple)) or not trajectory:
        raise ValueError("Generated trajectory required")
    for record in trajectory:
        if (
            not isinstance(record, (list, tuple))
            or len(record) != 2
            or type(record[0]) is not int
            or not 0 <= record[0] <= step + 1
        ):
            raise ValueError("Invalid or future trajectory")
        _text(record[1])
    if trajectory[-1][1] != location:
        raise ValueError("Trajectory and location disagree")
    visits = row["visited_locations"]
    if (
        not isinstance(visits, dict)
        or not visits
        or location not in visits
        or any(
            not isinstance(k, str) or type(v) is not int or v <= 0
            for k, v in visits.items()
        )
    ):
        raise ValueError("Invalid visit counts")
    if hri:
        if row["motivation"] is not None:
            raise ValueError("This initialization has no measured motivation")
        records = row["criminal_record"]
        if not isinstance(records, (list, tuple)):
            raise ValueError("Invalid criminal record")
        for record in records:
            if (
                not isinstance(record, (list, tuple))
                or len(record) != 2
                or type(record[0]) is not int
                or not -1 <= record[0] < step
            ):
                raise ValueError("Invalid or future crime record")
            _text(record[1])


def build_case(encounter, environment):
    """Accept only whitelisted pre-decision facts, with explicit provenance."""
    if not isinstance(encounter, dict) or set(encounter) != {
        "step",
        "criminal",
        "targets",
        "police_count",
        "current_location",
    }:
        raise ValueError("Unexpected encounter fields")
    if not isinstance(environment, dict) or set(environment) != set(ENV_FIELDS):
        raise ValueError("Unexpected environment fields")
    step, location = encounter["step"], encounter["current_location"]
    if type(step) is not int or step < 0:
        raise ValueError("Invalid step")
    for field in ("cbg_id", "description"):
        _text(environment[field])
    if location != environment["cbg_id"]:
        raise ValueError("Encounter and environment disagree")
    for field in set(ENV_FIELDS) - {"cbg_id", "description"}:
        value = environment[field]
        if value is None and field not in {"safety_score", "poi_count"}:
            continue
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid or (field in {"safety_score", "poverty_ratio"} and value > 1):
            raise ValueError("Invalid environment number")
    if type(environment["poi_count"]) is not int:
        raise ValueError("Invalid POI count")
    if type(encounter["police_count"]) is not int or encounter["police_count"] < 0:
        raise ValueError("Invalid police count")
    _person(encounter["criminal"], location, step, hri=True)
    targets = encounter["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("At least one encountered target required")
    for target in targets:
        _person(target, location, step)
    ids = [target["agent_id"] for target in targets]
    if len(set(ids)) != len(ids) or encounter["criminal"]["agent_id"] in ids:
        raise ValueError("Invalid target identities")
    facts = {
        "person": encounter["criminal"],
        "targets": targets,
        "environment": environment,
        "encounter": {
            "step": step,
            "current_location": location,
            "police_count": encounter["police_count"],
            "source": "EPR-generated co-location, not an observed real-world encounter",
        },
        "action_domain": "abstain or commit a simulated criminal act against a listed resident; no tactics or operational detail.",
    }
    # JSON freeze also normalizes source tuple histories and rejects nonfinite data.
    facts = json.loads(json.dumps(facts, allow_nan=False))
    return {
        "case_id": f"{step}:{encounter['criminal']['agent_id']}",
        "profile": "source",
        "facts": facts,
        "facts_sha256": _sha(facts),
        "target_ids": ids,
    }
