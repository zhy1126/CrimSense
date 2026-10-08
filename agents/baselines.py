"""Single-stage manuscript baselines; random control is handled by the runner."""

from copy import deepcopy
from . import templates
from .sat import facts, exact
from simulation.io import dumps


def baseline_job(case, method, city, *, profile=None):
    if method not in ("crimemind", "plain_llm"):
        raise ValueError("Unknown LLM baseline")
    f = facts(case, profile)
    f["city_context"] = templates.city_context(city)
    f["eligible_target_ids"] = list(case["target_ids"])
    if not f["eligible_target_ids"]:
        raise ValueError("No targets: skip the model")
    system = templates.render(method, PLAIN_FACTS=dumps(f), FACTS=dumps(f))

    def validate(v):
        if not isinstance(v, dict) or type(v.get("status")) is not bool:
            raise ValueError("An exact boolean status is required")
        exact(v, {"status", "reasoning"} | ({"objective_id"} if v["status"] else set()))
        if not isinstance(v["reasoning"], str) or not v["reasoning"].strip():
            raise ValueError("A brief rationale is required")
        if v["status"] and (
            not isinstance(v["objective_id"], str)
            or v["objective_id"] not in case["target_ids"]
        ):
            raise ValueError("Only supplied targets may be selected")
        return deepcopy(v)

    return dict(
        stage="choice",
        body=dict(
            model="Qwen2.5-7B-Instruct",
            temperature=1.0,
            max_tokens=512,
            messages=[dict(role="user", content=system)],
        ),
        validate=validate,
    )
