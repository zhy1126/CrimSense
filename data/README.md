# Data access

Simulation inputs are supplied locally under `data/external/`, which Git ignores. Configure their paths in `configs/city.example.json`.

- **City inputs:** CBG boundaries, environments, points of interest, observed crime counts, and initial agent profiles. Required formats are listed in [Data and protocols](../docs/data-and-protocols.md). Map loading is implemented in `simulation/world.py`; map files are external inputs.
- **Moral-response data:** Request access through the [ISRD4 source record](https://doi.org/10.5281/zenodo.19596719). Its access conditions limit use to attributed, non-commercial research and prohibit redistributing or reposting the dataset. This release does not distribute survey records or derived frequency tables. Obtain permission from the data custodian before sharing derived data.

For authorized local use, `patterns` points to a JSON object containing a `patterns` list. Each row contains integer responses (1–4) for `prosoc3`, `prosoc5`, `prosoc7`, and `prosoc8`, plus a positive integer frequency `n`. Keep complete four-item responses and count each distinct combination; propensity groups are computed by `agents/profiles.py`. City-level spatial validation and intervention summaries do not use these frequencies; omit the `patterns` entry from their manifests if the file is unavailable.

The example configuration does not exclude agents used in earlier experiments. New person–environment cohorts are sampled from the supplied actor population.
