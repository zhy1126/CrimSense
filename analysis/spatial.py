"""Spatial definitions retained from the reported experiments."""

import numpy as np


def evaluate_counts(predicted, observed, domain):
    keys = sorted(domain)
    if (
        len(keys) < 5
        or len(set(keys)) != len(keys)
        or any(not isinstance(k, str) for k in keys)
    ):
        raise ValueError("Need at least five unique domain identifiers")
    for values in (predicted, observed):
        if not isinstance(values, dict) or any(
            not isinstance(k, str) or type(v) is not int or v < 0
            for k, v in values.items()
        ):
            raise ValueError("Counts must be nonnegative integers")
    q = np.array([predicted.get(k, 0) for k in keys], float)
    p = np.array([observed.get(k, 0) for k in keys], float)
    if p.sum() <= 0:
        raise ValueError("Observed domain has no events")
    predicted_total = sum(predicted.values())
    inside = int(q.sum())
    observed_total = int(p.sum())
    h = int(len(keys) * 0.2)
    subset_count = max(2, int(len(keys) * 0.3))
    metrics = {"JSD": None, "RMSE": None, "HR@1": None, "HR@1.5": None, "HR@2": None}
    result = dict(
        status="defined" if inside else "undefined_no_predicted_events",
        domain=keys,
        crime_count_total=predicted_total,
        crime_count_in_domain=inside,
        crime_count_outside_domain=predicted_total - inside,
        observed_count_in_domain=observed_total,
        hotspot_count=h,
        hotspot_definition="Top floor(0.2*N) observed-count CBGs; lexical tie-breaking",
        full=dict(
            metrics,
            n=len(keys),
            definition="Base-2 Jensen-Shannon divergence and globally normalized spatial-share RMSE over the whole fixed domain",
        ),
        best30=dict(
            metrics,
            n=subset_count,
            definition="Author expression: best absolute-share-error 30%, natural-log square root, epsilon 1e-10, global normalization retained",
        ),
        hr_domain="Full fixed domain for both tables; HR is not evaluated on the selected 30%",
    )
    if not inside:
        return result
    p /= p.sum()
    q /= q.sum()
    mid = (p + q) / 2
    posp = p > 0
    posq = q > 0
    js = 0.5 * np.sum(p[posp] * np.log2(p[posp] / mid[posp])) + 0.5 * np.sum(
        q[posq] * np.log2(q[posq] / mid[posq])
    )
    result["full"].update(
        JSD=float(max(0, js)), RMSE=float(np.sqrt(np.mean((q - p) ** 2)))
    )
    selected = np.argsort(abs(q - p))[:subset_count]
    ps = p[selected] + 1e-10
    qs = q[selected] + 1e-10
    ms = (ps + qs) / 2
    selected_js = 0.5 * np.sum(ps * np.log(ps / ms)) + 0.5 * np.sum(
        qs * np.log(qs / ms)
    )
    result["best30"].update(
        JSD=float(np.sqrt(max(0, selected_js))),
        RMSE=float(np.sqrt(np.mean((q[selected] - p[selected]) ** 2))),
        selected_cbgs=[keys[i] for i in selected],
    )
    real = set(sorted(keys, key=lambda k: (-observed.get(k, 0), k))[:h])
    rank = sorted(keys, key=lambda k: (-predicted.get(k, 0), k))
    for name, factor in (("HR@1", 1), ("HR@1.5", 1.5), ("HR@2", 2)):
        score = len(real & set(rank[: int(h * factor)])) / h
        result["full"][name] = result["best30"][name] = score
    return result
