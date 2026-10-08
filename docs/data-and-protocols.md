# Data and protocols

## Input manifest

`configs/city.example.json` defines one city. Use `chicago`, `dallas`, or `los_angeles` as the city identifier. Files are JSON; no pickle caches are executed.

| File | Required content |
| --- | --- |
| `environments` | Rows with `cbg_id`, `description`, `safety_score`, `population`, `per_capita_income`, `poverty_ratio`, `housing_value`, `poi_count` |
| `pois` | OSM rows with `osm_id`, `cbg_id`, `lon`, `lat`; counts must match each environment row |
| `geometries` | Mapping from CBG identifiers to full WGS84 Polygon/MultiPolygon GeoJSON objects |
| `residents` | 4,000 source rows: `agent_id`, `residence`, `gender`, `race`, `income_level` |
| `actors` | 1,000 focal source rows: `agent_id`, `residence`, `gender`, `race` |
| `police` | 500 source rows: `agent_id`, `residence` |
| `config` | The reviewed EPR configuration, either its `epr_model` mapping or full source configuration |
| `observed` | Nonnegative integer crime counts keyed by CBG; required for city-level spatial validation |
| `domain` | Fixed list of CBG identifiers for city-level spatial validation |
| `patterns` | Locally supplied four-item response-pattern counts; see [Data access](../data/README.md) |
| `excluded_actors` | Optional local list of source actor IDs excluded before person–environment cohort sampling; omitted by default |

Missing ACS estimates remain null. Preserve leading zeros in Los Angeles identifiers. Its movement map contains 1,998 CBGs, while its fixed evaluation domain contains 1,997. Do not infer the evaluation domain from positive-count areas. Chicago's domain contains 1,152 CBGs and Dallas's 612.

The city bundle and source classes originate from the [CrimeMind repository](https://anonymous.4open.science/r/CrimeMind-EB3E/) and the project's reviewed public-data reconstruction. They are not bundled here. Obtain the matching input bundle from the project owner; an arbitrary new upstream checkout may not match. Four upstream class files are checked against recorded SHA-256 hashes in `simulation/world.py` before their definitions are loaded. Original source/data terms remain applicable; this repository does not relicense them.

The reconstruction retains the original experiments' EPSG:32616 distance calculation, including in Dallas and Los Angeles. This inherited comparison convention differs from those cities' local UTM zones. Mobility uses source EPR exploration/preferential-return rules and independent deterministic streams per agent and step. Police initially patrol uniformly across CBGs.

## Profiles and decisions

Four moral-response items (`prosoc3`, `prosoc5`, `prosoc7`, `prosoc8`) retain response categories 1–4. Mean responses below 2 map to high propensity; means from 2 up to 3 map to medium; means of at least 3 map to low. Synthetic profiles are sampled from locally supplied aggregate pattern frequencies. ISRD4 data and derived frequencies are not distributed with this release; see [Data access](../data/README.md).

City-level spatial validation uses source profiles without questionnaire assignment. Person–environment experiments vary moral profiles and safety/police inputs: low `(0.2, 0)`, medium `(0.5, 1)`, high `(0.8, 2)`. Each cohort uses the same actor identities across methods and conditions. Intervention experiments fix profiles across policy branches and world runs.

Perception returns an allowed action set. Guardianship is assessed separately and cannot read the perception result; in intervention runs it also excludes the personal profile. Commitment receives the retained alternatives and validated guardianship score. Target selection occurs only after commitment and can select only a retained, co-located target. No-target steps bypass the model. The three ablations remove the action filter, replace guardianship with the controlled safety score, or merge commitment and target selection. The merged variant exposes target information earlier and is not a pure comparison of stage count.

The current CrimeMind-style baseline implements the RAT prompt in the supplement. Its new runs are distinct from the original CrimeMind paper's published benchmark values. Published values should be cited from that paper rather than relabeled as outputs of this implementation.

## Interventions

All branches replay the same pre-intervention history. Activation is at zero-based step 25 (paper step 26). Target selection uses zero-based steps 5–19, ranks CBGs by crime count, and takes the shortest prefix covering at least 30% of eligible opportunities, capped at 100 CBGs.

- Hotspot policing reallocates 100 existing officers; total police population stays 500.
- Environmental improvement adds 0.2 to target-area safety, capped at 1.
- Resident cooperation selects 60% of existing residents whose homes are in selected areas. A participating witness can coordinate with up to two nearby participants within 1,000 m for next-step guardianship. The victim is excluded.
- Joint intervention combines these doses. Responses and movement remain explicit world updates.

No-intervention and intervention runs also retain the baseline witness/reporting rules and next-step police assistance, capped at 25 responding officers. The code in `simulation/policies.py`, `residents.py`, and `joint.py` is retained from the final research implementation; only imports were reorganized.

## Metrics and versioning

Let N be all focal actor-steps, Q eligible opportunities, K opportunities retaining crime, and C validated crime choices. The reported rates are `100*C/N`, `100*C/Q`, `100*K/Q`, and `100*C/K`. Undefined zero-denominator rates remain null.

Person–environment analysis uses opportunity rates. It reports the signed environmental contrast `ΔE(p) = r(p, high) − r(p, low)`, interaction `I = ΔE(high) − ΔE(low)`, and the reduction-oriented manuscript contrast `D = −I`. Actor bootstrap weights are shared across methods and all P/E conditions within a city/cohort. Equal-city summaries average city-specific rates and contrasts. Cohorts are not independent world repetitions.

Intervention analysis compares post-intervention counts/rates with control within each world and averages those paired differences across runs within a city. Adjacent areas share a polygon boundary with selected areas; selected, adjacent, and rest-of-city counts sum to the citywide count. Standard deviations use `ddof=1`; one-run SDs are null.

Spatial evaluation retains full-domain base-2 JSD, normalized-share RMSE, and HR@1/1.5/2. The legacy best-30% comparison is reported separately with its original natural-log square-root expression; it is not the full-domain JSD.

Each run records the experiment settings needed for analysis. The current prompt version is `manuscript-2026-10-08`; historical paper results predate these editorial prompt changes. Rerunning this release creates a new experiment version. Old results and request ledgers should remain in the research archive rather than be overwritten.
