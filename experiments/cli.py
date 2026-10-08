"""Run city-level validation, person–environment experiments, and policy evaluation."""

import argparse
from pathlib import Path
import os
from agents.pipeline import METHODS
from simulation.inputs import Inputs
from simulation.client import Client, ContextLimit
from simulation.io import write, read
from experiments.run import run_city, run_pe, run_interventions
from analysis.behavior import pe_summary, policy_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="CrimSense experiments and analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "city": "Evaluate simulated city-level crime distributions",
        "pe": "Compare person–environment responses and decision-stage ablations",
        "interventions": "Evaluate behavioral and spatial effects of crime-prevention policies",
    }
    for name, description in descriptions.items():
        p = sub.add_parser(name, help=description, description=description)
        p.add_argument("--inputs", required=True)
        p.add_argument("--output", required=True, help="New output directory")
        p.add_argument(
            "--tokenizer", default=os.environ.get("CRIMSENSE_TOKENIZER_ROOT")
        )
        p.add_argument("--workers", type=int, default=16)
        if name in ("city", "interventions"):
            p.add_argument("--seeds", nargs="+", type=int, required=True)
        if name == "city":
            p.add_argument(
                "--methods",
                nargs="+",
                choices=("crimsense", "crimemind", "plain_llm"),
                default=["crimsense"],
            )
        if name == "pe":
            p.add_argument("--cohort-seed", type=int, required=True)
            p.add_argument(
                "--methods",
                nargs="+",
                choices=METHODS,
                default=["abm_random", "plain_llm", "crimemind", "crimsense"],
            )
        if name == "interventions":
            p.add_argument("--profile-seed", type=int, required=True)
    p = sub.add_parser("summarize-pe")
    p.add_argument("runs", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--draws", type=int, default=4000)
    p = sub.add_parser("summarize-interventions")
    p.add_argument("run")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("plot")
    p.add_argument("kind", choices=("pe", "interventions"))
    p.add_argument("summary")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize-pe":
        write(args.output, pe_summary(args.runs, draws=args.draws))
        return
    if args.command == "plot":
        from analysis.figures import plot_pe, plot_interventions

        (plot_pe if args.kind == "pe" else plot_interventions)(
            read(args.summary), args.output
        )
        return
    inputs = Inputs(args.inputs)
    if args.command == "summarize-interventions":
        write(
            args.output,
            policy_summary(
                args.run,
                geometries={k: v["shapely_lnglat"] for k, v in inputs.map.aois.items()},
            ),
        )
        return
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if not args.tokenizer:
        parser.error("Set --tokenizer or CRIMSENSE_TOKENIZER_ROOT")
    if (len(inputs.actors), len(inputs.residents), len(inputs.police)) != (
        1000,
        4000,
        500,
    ):
        parser.error(
            "Paper inputs require 1,000 focal agents, 4,000 residents and 500 police"
        )
    if args.command == "city" and (inputs.observed is None or inputs.domain is None):
        parser.error(
            "City-level validation requires observed crime counts and a fixed evaluation domain"
        )
    with Client(context_check=ContextLimit(args.tokenizer)) as client:
        if args.command == "city":
            run_city(
                inputs,
                args.output,
                client,
                seeds=args.seeds,
                methods=args.methods,
                workers=args.workers,
            )
        elif args.command == "pe":
            run_pe(
                inputs,
                args.output,
                client,
                cohort_seed=args.cohort_seed,
                methods=args.methods,
                workers=args.workers,
            )
        else:
            run_interventions(
                inputs,
                args.output,
                client,
                seeds=args.seeds,
                profile_seed=args.profile_seed,
                workers=args.workers,
            )


if __name__ == "__main__":
    main()
