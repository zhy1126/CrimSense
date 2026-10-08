"""Shared decision stages for the full model, baselines, and three ablations."""

from copy import deepcopy
import json
from .sat import SAT, exact
from .baselines import baseline_job
from . import templates
from simulation.io import dumps

SAT_METHODS = (
    "crimsense",
    "no_action_filter",
    "rule_guardianship",
    "merged_commit_target",
)
METHODS = ("crimsense", "crimemind", "plain_llm", "abm_random") + SAT_METHODS[1:]


def merged_job(sat, case, prior, profile):
    job = sat.job(case, "commit", prior, profile=profile)
    message = job["body"]["messages"][0]["content"]
    start, end = message.index("JSON Output"), message.index("SAT Conditional Controls")
    message = (message[:start] + message[end:]).replace(
        "• Do not choose a target at this stage.\n", ""
    )
    # Only the separation and output contract change; retain commitment instructions.
    reminder = templates.text("shared_temptation")
    message += "\n" + reminder + "\n" + templates.text("merged_output")
    actions = prior["considered_action_ids"]
    targets = [a for a in actions if a != "abstain"]
    payload = json.loads(job["body"]["messages"][-1]["content"])
    payload.update(decision_stage="merged", target_action_ids=targets)
    payload["visible_facts"]["targets"] = [
        {k: r[k] for k in ("agent_id", "gender", "race")}
        for r in case["facts"]["targets"]
        if "offend:" + r["agent_id"] in targets
    ]
    job["stage"] = "merged"
    job["body"]["messages"][0]["content"] = message
    job["body"]["messages"][-1]["content"] = dumps(payload)
    job["body"]["response_format"] = {"type": "json_object"}

    def validate(v):
        exact(v, ("status", "action_id"))
        if type(v["status"]) is not bool:
            raise ValueError("Merged choice needs boolean status")
        if v["status"]:
            if v["action_id"] not in targets:
                raise ValueError("Merged target outside retained alternatives")
        elif "abstain" not in actions or v["action_id"] is not None:
            raise ValueError("Merged abstention must be available and have null target")
        return deepcopy(v)

    job["validate"] = validate
    return job


def decide(
    case, method, city, experiment, client, key, *, profile=None, observations=None
):
    if method not in METHODS or method == "abm_random":
        raise ValueError("Random control is evaluated by the experiment runner")
    result = dict(crime=False, target=None, outcomes={})
    if method in ("crimemind", "plain_llm"):
        value = client.call(
            key + ":choice", baseline_job(case, method, city, profile=profile)
        )
        result.update(
            crime=value["status"],
            target=value.get("objective_id"),
            outcomes={"choice": value},
        )
        return result
    if experiment != "pe" and method != "crimsense":
        raise ValueError("Ablations are defined for the controlled PE experiment")
    sat = SAT(city, experiment)
    prior = {}
    for stage in ("perception", "guardianship"):
        if method == "no_action_filter" and stage == "perception":
            value = {
                "considered_action_ids": ["abstain"]
                + ["offend:" + a for a in case["target_ids"]]
            }
        elif method == "rule_guardianship" and stage == "guardianship":
            score = case["facts"]["environment"]["safety_score"]
            if score not in (0.2, 0.5, 0.8):
                raise ValueError(
                    "Rule guardianship requires a registered PE safety score"
                )
            value = {"Guardianship_Strength_Score": score}
        else:
            value = client.call(
                key + ":" + stage,
                sat.job(case, stage, profile=profile, observations=observations),
            )
        prior.update(value)
        result["outcomes"][stage] = value
    result.update(prior)
    result["consideration"] = any(
        a != "abstain" for a in prior["considered_action_ids"]
    )
    if method == "merged_commit_target":
        value = client.call(key + ":merged", merged_job(sat, case, prior, profile))
        result["outcomes"]["merged"] = value
        result["crime"] = value["status"]
        if value["status"]:
            result["target"] = value["action_id"].removeprefix("offend:")
    else:
        value = client.call(
            key + ":commit",
            sat.job(case, "commit", prior, profile=profile, observations=observations),
        )
        result["outcomes"]["commit"] = value
        result["crime"] = value["status"]
        if value["status"]:
            target = client.call(
                key + ":target",
                sat.job(
                    case, "target", prior, profile=profile, observations=observations
                ),
            )
            result["outcomes"]["target"] = target
            result["target"] = target["action_id"].removeprefix("offend:")
    return result
