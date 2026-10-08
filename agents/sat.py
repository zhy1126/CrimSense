"""One SAT implementation: action formation, guardianship, commitment, target.

Output validation owns the stage boundaries. Template text is loaded only from
agents/prompts; no historical adapter chain rewrites the emitted messages.
"""

from copy import deepcopy
import math
from agents import templates
from agents.profiles import answers, TEMPTATION
from simulation.io import dumps, digest

STAGES = ("perception", "guardianship", "commit", "target")
_KIND_MENU = [
    dict(action_id="abstain", description="Do not act against any listed resident."),
    dict(
        action_id="offend",
        description="Commit a general simulated criminal act against a listed resident.",
    ),
]


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected or missing response fields")


def strength(x):
    if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError("Guardianship must be a number in [0,1]")
    return x


def facts(case, profile=None, *, history_limit=None):
    f = deepcopy(case["facts"])
    p = f["person"]
    records = p.pop("criminal_record", [])
    p["simulated_crime_events"] = [row for row in records if row[0] >= 0]
    if profile is None:
        p["source_initialization"] = {
            "step_minus_one_records": [row for row in records if row[0] == -1],
            "meaning": "source initialization only, not observed prior offending",
        }
    f["targets"] = [
        {k: v[k] for k in ("agent_id", "gender", "race")} for v in f["targets"]
    ]
    if profile is not None:
        p.pop("source_initialization", None)
        p["questionnaire_responses"] = answers(profile)
        p["motivation"] = deepcopy(TEMPTATION)
    if history_limit is not None:
        recent = []
        for _, loc in reversed(p.get("historical_trajectory", [])):
            if loc not in recent:
                recent.append(loc)
            if len(recent) == history_limit:
                break
        summary = {"recent_limit": history_limit}
        for key in (
            "simulated_crime_events",
            "historical_trajectory",
            "total_trajectory",
        ):
            if key in p:
                summary[key + "_total"] = len(p[key])
                p[key] = p[key][-history_limit:]
        if "visited_locations" in p:
            summary["distinct_visited_locations"] = len(p["visited_locations"])
            p["visited_locations"] = {
                k: p["visited_locations"][k]
                for k in recent
                if k in p["visited_locations"]
            }
        p["history_summary"] = summary
    return f


def retained(prior, ids):
    if not isinstance(prior, dict):
        raise ValueError("Validated earlier stages required")
    a = prior.get("considered_action_ids")
    allowed = ["abstain"] + ["offend:" + x for x in ids]
    if (
        not isinstance(a, list)
        or not a
        or len(a) != len(set(a))
        or any(x not in allowed for x in a)
    ):
        raise ValueError(
            "Retained actions must be a nonempty subset of the actual menu"
        )
    strength(prior.get("Guardianship_Strength_Score"))
    return list(a)


class SAT:
    def __init__(self, city, experiment="pe"):
        templates.city_context(city)
        if experiment not in ("city", "pe", "interventions"):
            raise ValueError("Unknown experiment")
        self.city, self.experiment = city, experiment

    def job(self, case, stage, prior=None, *, profile=None, observations=None):
        if stage not in STAGES:
            raise ValueError("Unknown stage")
        if stage in ("perception", "guardianship") and prior is not None:
            raise ValueError("Independent inputs required")
        if self.experiment != "city" and profile is None:
            raise ValueError("Controlled experiments need a moral-response profile")
        ids = case["target_ids"]
        if not ids:
            raise ValueError("No opportunity: skip LLM decision")
        f = facts(
            case,
            profile,
            history_limit=5 if self.experiment == "interventions" else None,
        )
        original = deepcopy(f)
        if stage == "guardianship" and self.experiment == "interventions":
            f.pop("person", None)
        if stage == "commit" and self.experiment != "city":
            p = f["person"]
            env = f["environment"]
            enc = f["encounter"]
            f = {
                "action_domain": f["action_domain"],
                "person": {
                    k: p[k]
                    for k in (
                        "questionnaire_responses",
                        "motivation",
                        "simulated_crime_events",
                    )
                },
                "environment": {k: env[k] for k in ("safety_score", "description")},
                "encounter": {"police_count": enc["police_count"]},
            }
        if observations is not None:
            f["local_observations"] = deepcopy(observations)
        menu = deepcopy(_KIND_MENU)
        if stage in ("commit", "target") and self.experiment != "interventions":
            menu = [menu[0]] + [
                dict(
                    action_id="offend:" + x,
                    description="Commit a general simulated criminal act against resident "
                    + x
                    + ".",
                )
                for x in ids
            ]
        payload = dict(
            decision_protocol=templates.VERSION,
            decision_stage=stage,
            objective_menu=menu,
            visible_facts=f,
        )
        values = dict(
            SAFETY_SCORE=original["environment"]["safety_score"],
            POLICE_COUNT=original["encounter"]["police_count"],
            TEMPTATION_REMINDER=(
                templates.text("shared_temptation") if profile is not None else ""
            ),
        )
        if stage in ("commit", "target"):
            actions = retained(prior, ids)
            kinds = (["abstain"] if "abstain" in actions else []) + (
                ["offend"] if any(x != "abstain" for x in actions) else []
            )
            payload.update(
                considered_action_ids=actions,
                perceived_guardianship_strength=prior["Guardianship_Strength_Score"],
            )
            values["GUARDIANSHIP"] = prior["Guardianship_Strength_Score"]
        if stage == "perception":
            system = templates.render("action_formation", **values)
            if profile is None:
                # S5 supplies the source-city role; no questionnaire profile is invented.
                system = "Perception Task\n" + system.split("Perception Task\n", 1)[1]
                system = system.replace(
                    "The supplied motivation is an explicit\nexperimental scenario condition, not a measured personal trait.",
                    "The supplied motivation remains unknown.",
                )

            def validate(v):
                exact(v, ["personal_action_set"])
                x = v["personal_action_set"]
                if x not in ("abstain_only", "both", "offend_only"):
                    raise ValueError("Unknown personal action set")
                return {
                    "considered_action_ids": (["abstain"] if x != "offend_only" else [])
                    + (["offend:" + i for i in ids] if x != "abstain_only" else [])
                }

            props = {
                "personal_action_set": {
                    "type": "string",
                    "enum": ["abstain_only", "both", "offend_only"],
                }
            }
        elif stage == "guardianship":
            system = templates.text("guardianship")

            def validate(v):
                exact(v, ["Guardianship_Strength_Score"])
                return {
                    "Guardianship_Strength_Score": strength(
                        v["Guardianship_Strength_Score"]
                    )
                }

            props = {
                "Guardianship_Strength_Score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                }
            }
        elif stage == "commit":
            payload["available_kinds"] = kinds
            system = templates.render("commitment", **values)
            if profile is None:
                system = "Choice Task\n" + system.split("Choice Task\n", 1)[1]

            def validate(v):
                exact(v, ["status"])
                if (
                    type(v["status"]) is not bool
                    or ("offend" if v["status"] else "abstain") not in kinds
                ):
                    raise ValueError("Choice outside the retained action kinds")
                return deepcopy(v)

            props = {"status": {"type": "boolean"}}
            if len(kinds) == 1:
                props["status"]["enum"] = [kinds[0] == "offend"]
        else:
            targets = [x for x in actions if x != "abstain"]
            if not targets:
                raise ValueError("No retained target after commitment")
            payload["target_action_ids"] = targets
            system = templates.text("target_selection")

            def validate(v):
                exact(v, ["action_id"])
                if not isinstance(v["action_id"], str) or v["action_id"] not in targets:
                    raise ValueError("Target must be an actually retained resident")
                return deepcopy(v)

            props = {"action_id": {"type": "string", "enum": targets}}
        if self.experiment == "city":
            frame = templates.render(
                "urban_context",
                CITY_CONTEXT=templates.city_context(self.city),
                ASSESSMENT_SENTENCE="Assess the setting using all local evidence, including police and the scene description.",
                STAGE_ENDING={
                    "perception": "Report only your consideration set; environmental scoring and choice are separate tasks.",
                    "guardianship": "Report only your environmental guardianship assessment.",
                    "commit": "Choose only among the retained action kinds.",
                    "target": "Select exactly one retained target after the offense commitment.",
                }[stage],
                **values,
            )
            system = frame + "\n\n" + system
        body = dict(
            model="Qwen2.5-7B-Instruct",
            temperature=1.0,
            max_tokens=600 if stage in ("perception", "guardianship") else 200,
            messages=[
                dict(role="system", content=system),
                dict(role="user", content=dumps(payload)),
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sat_" + stage,
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": props,
                        "required": list(props),
                        "additionalProperties": False,
                    },
                },
            },
        )
        if stage == "commit" and len(kinds) == 2:
            body["response_format"] = {"type": "json_object"}
        return dict(stage=stage, body=body, validate=validate)
