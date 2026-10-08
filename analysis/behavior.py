"""Behavioral rates, paired actor-cluster PE intervals, and policy summaries."""

from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from agents.profiles import LEVELS
from simulation.io import read
from experiments.run import POLICIES


def rate(n, d):
    return 100 * n / d if d else None


def counts(rows):
    n = len(rows)
    q = sum(r["opportunity"] for r in rows)
    c = sum(r["crime"] for r in rows)
    if c > q:
        raise ValueError("Crime without opportunity")
    has_stages = all("consideration" in r for r in rows if r["opportunity"])
    k = sum(r.get("consideration", False) for r in rows) if has_stages else None
    if k is not None and not c <= k <= q:
        raise ValueError("Invalid stage counts")
    return dict(
        N=n,
        Q=q,
        K=k,
        C=c,
        overall_rate=rate(c, n),
        opportunity_rate=rate(c, q),
        consideration_rate=rate(k, q) if k is not None else None,
        conditional_rate=rate(c, k) if k is not None else None,
    )


def _ratio(c, q):
    return np.divide(c, q, out=np.full_like(c, np.nan, dtype=float), where=q != 0) * 100


def _stat(point, draws):
    samples = np.asarray(draws)
    samples = samples[np.isfinite(samples)]
    return dict(
        mean=float(point) if np.isfinite(point) else None,
        ci95=np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) else None,
        valid_draws=len(samples),
    )


def _pe_summary(point, boot, methods):
    result = {}
    for i, method in enumerate(methods):
        delta = point[i, :, 2] - point[i, :, 0]
        samples = boot[:, i, :, 2] - boot[:, i, :, 0]
        result[method] = dict(
            cells={
                p + "__" + e: _stat(point[i, pi, ei], boot[:, i, pi, ei])
                for pi, p in enumerate(LEVELS)
                for ei, e in enumerate(LEVELS)
            },
            delta_E={
                p: _stat(delta[pi], samples[:, pi]) for pi, p in enumerate(LEVELS)
            },
            I=_stat(delta[2] - delta[0], samples[:, 2] - samples[:, 0]),
            D=_stat(delta[0] - delta[2], samples[:, 0] - samples[:, 2]),
        )
    return result


def pe_city(output, *, draws=4000, bootstrap_seed=2026092211):
    root = Path(output)
    reg = read(root / "experiment.json")
    cfg = reg["config"]
    if cfg["experiment"] != "pe":
        raise ValueError("Expected a PE run")
    cohorts = cfg["cohorts"]
    methods = cfg["methods"]
    steps = cfg["steps"]
    actors = [a for cohort in cohorts for a in cohort["actors"]]
    if len(actors) != len(set(actors)):
        raise ValueError("Duplicate actor across cohorts")
    index = {a: i for i, a in enumerate(actors)}
    totals = np.zeros((len(actors), len(methods), 3, 3, 2))
    for cohort in cohorts:
        for mi, method in enumerate(methods):
            for pi, p in enumerate(LEVELS):
                for ei, e in enumerate(LEVELS):
                    for step in range(steps):
                        path = (
                            root
                            / str(cohort["index"])
                            / method
                            / p
                            / e
                            / f"{step:03}.json"
                        )
                        rows = read(path)["rows"]
                        if Counter(r["actor"] for r in rows) != Counter(
                            cohort["actors"]
                        ):
                            raise ValueError("Missing or duplicate PE actor-step")
                        for row in rows:
                            if (
                                row["step"],
                                row["method"],
                                row["p"],
                                row["e"],
                                row["cohort"],
                            ) != (step, method, p, e, cohort["index"]):
                                raise ValueError("PE condition mismatch")
                            if (
                                type(row["crime"]) is not bool
                                or type(row["opportunity"]) is not bool
                                or row["crime"] > row["opportunity"]
                            ):
                                raise ValueError("Invalid PE decision")
                            totals[index[row["actor"]], mi, pi, ei] += [
                                row["opportunity"],
                                row["crime"],
                            ]
    summed = totals.sum(axis=0)
    point = _ratio(summed[..., 1], summed[..., 0])
    rng = np.random.default_rng(bootstrap_seed)
    # One actor weight shared by every method, P and E condition; stratify by cohort.
    weights = np.concatenate(
        [
            rng.multinomial(
                len(c["actors"]),
                np.ones(len(c["actors"])) / len(c["actors"]),
                size=draws,
            )
            for c in cohorts
        ],
        axis=1,
    )
    samples = np.tensordot(weights, totals, axes=(1, 0))
    boot = _ratio(samples[..., 1], samples[..., 0])
    result = dict(
        city=reg["city"],
        methods=_pe_summary(point, boot, methods),
        draws=draws,
        bootstrap_seed=bootstrap_seed,
        denominator="opportunity actor-steps",
    )
    return result, point, boot, methods


def pe_summary(outputs, *, draws=4000, bootstrap_seed=2026092211):
    if type(draws) is not int or draws < 2:
        raise ValueError("Need at least two bootstrap draws")

    def protocol(reg):
        cfg = reg["config"]
        if cfg["experiment"] != "pe":
            raise ValueError("Expected a PE run")
        return dict(
            prompt_version=reg["prompt_version"],
            mode=reg["execution_mode"],
            steps=cfg["steps"],
            methods=cfg["methods"],
            cohort_sizes=[len(c["actors"]) for c in cfg["cohorts"]],
        )

    registrations = [read(Path(o) / "experiment.json") for o in outputs]
    if not registrations:
        raise ValueError("No cities supplied")
    shared = protocol(registrations[0])
    if any(protocol(r) != shared for r in registrations[1:]):
        raise ValueError(
            "Incompatible PE prompt version, execution mode or design across cities"
        )
    cities = {}
    points = []
    boots = []
    methods = None
    for i, output in enumerate(outputs):
        result, point, boot, names = pe_city(
            output, draws=draws, bootstrap_seed=bootstrap_seed + i
        )
        if result["city"] in cities:
            raise ValueError("Only one PE result per city")
        if methods is not None and names != methods:
            raise ValueError("Methods must match across cities")
        cities[result["city"]] = result
        methods = names
        points.append(point)
        boots.append(boot)
    if not cities:
        raise ValueError("No cities supplied")
    return dict(
        protocol=shared,
        by_city=cities,
        equal_city_average=_pe_summary(
            np.mean(points, axis=0), np.mean(boots, axis=0), methods
        ),
        unit="percentage points / percent",
        interaction="I=(r_HH-r_HL)-(r_LH-r_LL); D=-I",
        uncertainty="Paired actor-cluster bootstrap within each city/cohort, conditional on fixed worlds",
    )


def mean_sd(values):
    if any(v is None for v in values):
        return dict(mean=None, sd=None, n=len(values))
    return dict(
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        n=len(values),
    )


def policy_summary(output, *, geometries, activation=25, window=5):
    root = Path(output)
    reg = read(root / "experiment.json")
    cfg = reg["config"]
    if cfg["experiment"] != "interventions":
        raise ValueError("Expected an intervention run")
    if cfg.get("parameters"):
        activation = cfg["parameters"].get("activation_step", activation)
    runs = []
    for seed in cfg["seeds"]:
        targets = set(read(root / str(seed) / "target_plan.json")["targets"])
        if not targets <= geometries.keys():
            raise ValueError("Missing selected-area geometry")
        # Boundary adjacency; selected and adjacent sets remain disjoint.
        adjacent = {
            g
            for g, geom in geometries.items()
            if g not in targets and any(geom.touches(geometries[t]) for t in targets)
        }
        zones = {
            "selected": targets,
            "adjacent": adjacent,
            "rest": set(geometries) - targets - adjacent,
            "citywide": set(geometries),
        }
        branches = {}
        for arm in POLICIES:
            records = []
            expected_actors = set(cfg["profiles"])
            for step in range(cfg["steps"]):
                rows = read(root / str(seed) / arm / f"{step:03}.json")["rows"]
                if (
                    len(rows) != len(expected_actors)
                    or {r["actor"] for r in rows} != expected_actors
                ):
                    raise ValueError("Incomplete policy actor-step records")
                if any(r["step"] != step for r in rows):
                    raise ValueError("Policy step mismatch")
                records.extend(rows)
            post = [r for r in records if r["step"] >= activation]
            cbg_counts = Counter(r["location"] for r in post if r["crime"])
            branches[arm] = dict(
                rates=counts(post),
                groups={p: counts([r for r in post if r["p"] == p]) for p in LEVELS},
                zones={
                    zone: sum(r["crime"] for r in post if r["location"] in ids)
                    for zone, ids in zones.items()
                },
                cbg_counts={g: cbg_counts[g] for g in geometries},
                windows=[
                    counts([r for r in records if start <= r["step"] < start + window])[
                        "overall_rate"
                    ]
                    for start in range(0, cfg["steps"], window)
                ],
            )
        control = branches["control"]
        for arm, b in branches.items():
            b["reduction_pp"] = (
                control["rates"]["overall_rate"] - b["rates"]["overall_rate"]
            )
            b["group_reductions"] = {
                p: control["groups"][p]["overall_rate"] - b["groups"][p]["overall_rate"]
                for p in LEVELS
            }
            b["spatial_change"] = {
                zone: b["zones"][zone] - control["zones"][zone] for zone in zones
            }
            b["cbg_change"] = {
                g: b["cbg_counts"][g] - control["cbg_counts"][g] for g in geometries
            }
            d = b["spatial_change"]
            if d["selected"] + d["adjacent"] + d["rest"] != d["citywide"]:
                raise ValueError("Spatial zones do not add up")
        runs.append(dict(seed=seed, branches=branches))
    summary = {}
    for arm in POLICIES:
        bs = [r["branches"][arm] for r in runs]
        summary[arm] = dict(
            rates={
                k: mean_sd([b["rates"][k] for b in bs])
                for k in (
                    "overall_rate",
                    "opportunity_rate",
                    "consideration_rate",
                    "conditional_rate",
                )
            },
            reduction_pp=mean_sd([b["reduction_pp"] for b in bs]),
            group_reductions={
                p: mean_sd([b["group_reductions"][p] for b in bs]) for p in LEVELS
            },
            spatial_change={
                z: mean_sd([b["spatial_change"][z] for b in bs])
                for z in ("selected", "citywide", "adjacent", "rest")
            },
            cbg_change={
                g: mean_sd([b["cbg_change"][g] for b in bs]) for g in geometries
            },
            windows=[
                mean_sd([b["windows"][i] for b in bs])
                for i in range(len(bs[0]["windows"]))
            ],
        )
    return dict(
        city=reg["city"],
        protocol=dict(
            prompt_version=reg["prompt_version"], execution_mode=reg["execution_mode"]
        ),
        runs=runs,
        summary=summary,
        activation_step=activation,
        window=window,
        uncertainty="Sample SD across runs within this city",
    )
