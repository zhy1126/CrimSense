"""One runner for city validation, controlled PE, ablations, and interventions."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import random

from agents.facts import build_case
from agents.pipeline import decide, METHODS, SAT_METHODS
from agents.profiles import assign_three, fixed_profiles, LEVELS
from agents import templates
from simulation.io import read, write, digest
from simulation.policies import PolicyWorld, build_target_plan
from simulation.residents import ResidentResponseWorld
from simulation.joint import JointPolicyWorld
from analysis.spatial import evaluate_counts

POLICIES = ("control", "hotspot", "environment", "residents", "joint")
SCORES = {"low": 0.2, "medium": 0.5, "high": 0.8}
PATROLS = {"low": 0, "medium": 1, "high": 2}


def start_run(inputs, output, config, mode):
    """Write experiment settings needed to interpret the resulting records."""
    path = Path(output)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Choose an empty output directory for a new experiment")
    write(
        path / "experiment.json",
        dict(
            city=inputs.city,
            config=config,
            prompt_version=templates.VERSION,
            execution_mode=mode,
        ),
    )


def decisions(
    world,
    city,
    method,
    experiment,
    client,
    namespace,
    step,
    *,
    profiles=None,
    overlay=None,
    e=None,
    p=None,
    cohort=None,
    workers=16,
):
    scenes = world.encounters(step)

    def one(encounter):
        actor = encounter["criminal"]["agent_id"]
        location = encounter["current_location"]
        env = deepcopy(
            overlay.environment(location)
            if overlay
            else world.map.environments[location]
        )
        if e is not None:
            env["safety_score"] = SCORES[e]
            encounter["police_count"] = PATROLS[e]
        profile = profiles[actor]["profile"] if profiles is not None else None
        group = profiles[actor]["p"] if profiles is not None else None
        obs = overlay.observations(location) if overlay else None
        row = dict(
            city=city,
            method=method,
            seed=world.seed,
            step=step,
            actor=actor,
            location=location,
            opportunity=bool(encounter["targets"]),
            crime=False,
            target=None,
            p=group,
            case_hash=digest(
                dict(
                    encounter=encounter,
                    environment=env,
                    observations=obs,
                    profile=profile,
                )
            ),
        )
        if e is not None:
            row.update(
                e=e,
                p=p,
                cohort=cohort,
                profile=profile,
                safety=SCORES[e],
                police_count=PATROLS[e],
            )
        if row["opportunity"]:
            case = build_case(encounter, env)
            if method == "abm_random":
                # The same actor-step draw is reused across all P and E conditions.
                rng = random.Random(
                    digest([city, world.seed, cohort, actor, step, "abm_random"])
                )
                row["crime"] = rng.random() < 0.5
                if row["crime"]:
                    row["target"] = rng.choice(case["target_ids"])
            else:
                row.update(
                    decide(
                        case,
                        method,
                        city,
                        experiment,
                        client,
                        f"{namespace}:{step}:{actor}",
                        profile=profile,
                        observations=obs,
                    )
                )
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, scenes))


def apply(world, rows, overlay=None):
    if overlay:
        overlay.finish_step(rows)
    else:
        for row in rows:
            if row["crime"]:
                world.record_crime(row["actor"], row["step"], row["location"])


def run_city(
    inputs, output, client, *, seeds, methods=("crimsense",), steps=50, workers=16
):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique seeds required")
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(m not in ("crimsense", "crimemind", "plain_llm") for m in methods)
    ):
        raise ValueError("Unsupported city method")
    start_run(
        inputs,
        output,
        dict(experiment="city", seeds=seeds, methods=methods, steps=steps),
        client.mode,
    )
    summaries = []
    for seed in seeds:
        for method in methods:
            world = inputs.world(seed)
            namespace = f"city:{inputs.city}:{seed}:{method}"
            all_rows = []
            for step in range(steps):
                world.move(step)
                rows = decisions(
                    world,
                    inputs.city,
                    method,
                    "city",
                    client,
                    namespace,
                    step,
                    workers=workers,
                )
                apply(world, rows)
                write(
                    Path(output) / str(seed) / method / f"{step:03}.json",
                    dict(rows=rows),
                )
                all_rows.extend(rows)
                print(
                    f"{inputs.city} / {method} / {seed}: step {step+1}/{steps}",
                    flush=True,
                )
            counts = dict(Counter(r["location"] for r in all_rows if r["crime"]))
            summary = dict(
                city=inputs.city,
                seed=seed,
                method=method,
                crimes=sum(counts.values()),
                counts=counts,
            )
            if inputs.observed is not None and inputs.domain is not None:
                summary["spatial"] = evaluate_counts(
                    counts, inputs.observed, inputs.domain
                )
            summaries.append(summary)
    write(Path(output) / "summary.json", summaries)
    return summaries


def cohorts(inputs, seed, size=100, count=3):
    pool = [
        r["agent_id"] for r in inputs.actors if r["agent_id"] not in inputs.excluded
    ]
    if len(pool) < size * count or len(set(pool)) != len(pool):
        raise ValueError("Need enough unique actors after the recorded exclusions")
    random.Random(seed).shuffle(pool)
    return [
        dict(
            index=i,
            seed=seed + i,
            actors=pool[i * size : (i + 1) * size],
            profiles=assign_three(
                inputs.patterns, pool[i * size : (i + 1) * size], seed + i
            ),
        )
        for i in range(count)
    ]


def run_pe(
    inputs,
    output,
    client,
    *,
    cohort_seed,
    methods=("abm_random", "plain_llm", "crimemind", "crimsense"),
    steps=10,
    cohort_size=100,
    cohort_count=3,
    workers=16,
):
    if len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Unique supported methods required")
    selected = cohorts(inputs, cohort_seed, cohort_size, cohort_count)
    start_run(
        inputs,
        output,
        dict(experiment="pe", methods=methods, steps=steps, cohorts=selected),
        client.mode,
    )
    for cohort in selected:
        for method in methods:
            for p in LEVELS:
                profiles = {
                    a: dict(p=p, profile=v) for a, v in cohort["profiles"][p].items()
                }
                for e in LEVELS:
                    world = inputs.world(cohort["seed"], cohort["actors"])
                    namespace = f'pe:{inputs.city}:{cohort["index"]}:{method}:{p}:{e}'
                    for step in range(steps):
                        world.move(step)
                        rows = decisions(
                            world,
                            inputs.city,
                            method,
                            "pe",
                            client,
                            namespace,
                            step,
                            profiles=profiles,
                            e=e,
                            p=p,
                            cohort=cohort["index"],
                            workers=workers,
                        )
                        apply(world, rows)
                        path = (
                            Path(output)
                            / str(cohort["index"])
                            / method
                            / p
                            / e
                            / f"{step:03}.json"
                        )
                        write(path, dict(rows=rows))
                    print(
                        f'{inputs.city} / cohort {cohort["index"]} / {method} / {p} / {e}: complete',
                        flush=True,
                    )


def make_policy(inputs, seed, arm, parameters=None):
    ready = {"worlds": {"sat": inputs.world(seed)}}
    if arm == "joint":
        return JointPolicyWorld(ready, seed, parameters)
    if arm == "residents":
        return ResidentResponseWorld(ready, "collective", seed, parameters)
    return PolicyWorld(ready, arm, seed, parameters)


def run_interventions(
    inputs,
    output,
    client,
    *,
    seeds,
    profile_seed,
    steps=50,
    workers=16,
    parameters=None,
):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unique seeds required")
    profiles = fixed_profiles(
        inputs.patterns, [r["agent_id"] for r in inputs.actors], profile_seed
    )
    config = dict(
        experiment="interventions",
        seeds=seeds,
        steps=steps,
        profile_seed=profile_seed,
        profiles=profiles,
        parameters=parameters,
    )
    start_run(inputs, output, config, client.mode)
    for seed in seeds:
        warm_rows = []
        warm_steps = {}
        plan = None
        for arm in POLICIES:
            overlay = make_policy(inputs, seed, arm, parameters)
            world = overlay.world
            activation = overlay.parameters["activation_step"]
            if steps <= activation:
                raise ValueError("Run must contain post-intervention steps")
            if plan is not None:
                overlay.configure(plan)
            for step in range(steps):
                if step == activation and arm == "control":
                    plan = build_target_plan(
                        warm_rows, world.source_rows["residents"], seed, parameters
                    )
                    overlay.configure(plan)
                    write(Path(output) / str(seed) / "target_plan.json", plan)
                overlay.move(step)
                shared = step < activation
                namespace = (
                    f'interventions:{inputs.city}:{seed}:{"warmup" if shared else arm}'
                )
                # Use the same realized warm-up decisions in every branch.
                if shared and arm != "control":
                    rows = deepcopy(warm_steps[step])
                else:
                    rows = decisions(
                        world,
                        inputs.city,
                        "crimsense",
                        "interventions",
                        client,
                        namespace,
                        step,
                        profiles=profiles,
                        overlay=overlay,
                        workers=workers,
                    )
                apply(world, rows, overlay)
                if shared and arm == "control":
                    warm_rows.extend(rows)
                    warm_steps[step] = deepcopy(rows)
                write(
                    Path(output) / str(seed) / arm / f"{step:03}.json",
                    dict(rows=rows),
                )
                print(
                    f"{inputs.city} / {arm} / {seed}: step {step+1}/{steps}", flush=True
                )
