"""ISRD response categories and the recorded synthetic profile sampling rules."""

import random

ITEMS = {
    "prosoc3": "Personal judgement about intentionally damaging another person's property.",
    "prosoc5": "Personal judgement about taking a small item from a shop without paying.",
    "prosoc7": "Personal judgement about intentionally injuring someone through an attack.",
    "prosoc8": "Personal judgement about taking money or possessions using violence or threats.",
}


LABELS = {1: "not wrong at all", 2: "a little wrong", 3: "wrong", 4: "very wrong"}


def answers(profile):
    if (
        not isinstance(profile, dict)
        or set(profile) != set(ITEMS)
        or any(type(v) is not int or v not in LABELS for v in profile.values())
    ):
        raise ValueError(
            "Exactly four valid morality answers required; no outcomes or extra traits"
        )
    return [
        dict(item=k, question=q, answer=LABELS[profile[k]]) for k, q in ITEMS.items()
    ]


LEVELS = PROPENSITY_LEVELS = ("low", "medium", "high")


def assign_three(data, ids, seed=42):
    """Map observed moral-answer patterns to three propensity bands."""
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Unique nonempty actor IDs required")
    pools = {p: [] for p in PROPENSITY_LEVELS}
    for row in data["patterns"]:
        profile = {k: row[k] for k in ITEMS}
        answers(profile)
        mean = sum(profile.values()) / len(profile)
        level = "high" if mean < 2.0 else ("medium" if mean < 3.0 else "low")
        pools[level].append((profile, int(row.get("n", 1))))
    if any(not pools[p] for p in PROPENSITY_LEVELS):
        raise ValueError("Observed answer patterns do not cover all propensity bands")
    result = {}
    for level in PROPENSITY_LEVELS:
        pairs = sorted(pools[level], key=lambda x: tuple(x[0][k] for k in ITEMS))
        rng = random.Random(f"{seed}:ISRD-morality-3x3:{level}")
        result[level] = {
            actor: dict(profile)
            for actor, profile in zip(
                sorted(ids),
                rng.choices(
                    [x for x, n in pairs], weights=[n for x, n in pairs], k=len(ids)
                ),
                strict=True,
            )
        }
    return result


def fixed_profiles(patterns, actors, seed):
    ids = sorted(actors)
    random.Random(str(seed) + ":fixed-cohort").shuffle(ids)
    assigned = assign_three(patterns, ids, seed)
    return {
        actor: dict(p=LEVELS[i % 3], profile=assigned[LEVELS[i % 3]][actor])
        for i, actor in enumerate(ids)
    }


TEMPTATION = dict(
    source="experimenter_assigned_synthetic_scenario",
    kind="situational_temptation",
    description="In this encounter the person notices an immediately attractive personal gain associated with the displayed general offending action. Abstaining leaves that gain unrealized. This is a momentary temptation, not an intention, moral acceptance, habitual criminal tendency, or certainty of acting. No particular offense, property, payment or tactic is specified.",
)
