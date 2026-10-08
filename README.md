# CrimSense

A theory-guided multi-agent crime simulator implementing Situational Action Theory through action formation, situational guardianship assessment, commitment, and target selection.

| Experiment | What it evaluates | Comparisons |
| --- | --- | --- |
| **City-Level Crime Distributions** | Whether simulated events reproduce observed crime concentrations | Chicago, Dallas, Los Angeles; JSD, RMSE, and hotspot recall |
| **Person–Environment Interaction** | How crime decisions respond to personal propensity and environmental protection | ABM-Random, Plain LLM, CrimeMind-style RAT baseline, and CrimSense; three propensity groups × three safety levels |
| **Decision-Stage Ablations** | How changes to individual decision stages affect person–environment responses | No action filter, rule guardianship, and merged commitment/target selection |
| **Intervention Effects** | How policies change crime decisions across propensity groups and locations | Control, hotspot policing, environmental improvement, resident cooperation, and joint intervention; three runs per city |

```text
agents/          Decision stages, baselines, profiles, and prompts
simulation/      City movement, interventions, and model calls
experiments/     Shared run entry points
analysis/        Metrics, summaries, and figure exports
configs/         City input example
data/            Input formats and access instructions
```

Prompts are maintained once in [`agents/prompts/`](agents/prompts), following the supplementary material supplied on October 8, 2026. These revised prompts define a new run version; the historical manuscript results were produced before the wording changes.

## Setup

Use Python 3.11+ on Linux or macOS. Install with `pip install -e .`. Obtain local inputs following [Data access](data/README.md) and [Data and protocols](docs/data-and-protocols.md), then create a manifest per city from `configs/city.example.json`. City maps and ISRD4-derived frequency data are supplied separately.

```bash
export CRIMSENSE_BASE_URL=https://your-model-server.example/v1
export CRIMSENSE_TOKENIZER_ROOT=/path/to/Qwen2.5-7B-Instruct-tokenizer
# Set CRIMSENSE_API_KEY only if the server requires authentication.
```

## Run experiments

### City-level crime distributions

```bash
python run_simulation.py city --inputs configs/chicago.json \
  --output outputs/chicago-city --seeds 42
```

### Person–environment interaction

```bash
python run_simulation.py pe --inputs configs/chicago.json \
  --output outputs/chicago-pe --cohort-seed 2026091701
```

### Decision-stage ablations

```bash
python run_simulation.py pe --inputs configs/chicago.json \
  --output outputs/chicago-ablation --cohort-seed 2026091701 \
  --methods crimsense no_action_filter rule_guardianship merged_commit_target
```

### Intervention effects

```bash
python run_simulation.py interventions --inputs configs/chicago.json \
  --output outputs/chicago-policy --seeds 2026092602 2026092603 2026092604 \
  --profile-seed 2026092602
```

These commands call the configured model server. Use a new output directory for each experiment. Repeat with the Dallas and Los Angeles manifests; their intervention world seeds are `2026100201 2026100202 2026100203`, with profile seed `2026092602`. Each city/intervention simulation has 50 steps. Person–environment experiments use three cohorts of 100 actors and 10 steps per condition.

## Analyze and export figures

```bash
python run_simulation.py summarize-pe outputs/chicago-pe outputs/dallas-pe outputs/la-pe \
  --output outputs/pe-summary.json
python run_simulation.py plot pe outputs/pe-summary.json --output outputs/pe-figures

python run_simulation.py summarize-interventions outputs/chicago-policy \
  --inputs configs/chicago.json --output outputs/chicago-policy-summary.json
python run_simulation.py plot interventions outputs/chicago-policy-summary.json \
  --output outputs/chicago-policy-figures
```

City-level validation writes spatial metrics to `summary.json`. Person–environment analysis uses paired actor-cluster bootstrap intervals and equal-city averaging. Intervention analysis reports means and sample standard deviations within each city. Figures export as individual PDFs with separate person–environment legends, distinct line styles, and hatched bars.
