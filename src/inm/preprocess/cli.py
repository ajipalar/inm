"""CLI entry point for the preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging

from inm.preprocess.config import load_config
from inm.preprocess import (
    parse_apms,
    build_spec_table,
    compute_correlations,
    fetch_sequences,
    add_sequences,
    run_hhblits,
    parse_hhr,
    filter_hits,
    fetch_structures,
    chain_mapping,
    compute_bsasa,
    build_reference,
    build_benchmark,
)

STEPS = {
    "parse-apms": parse_apms,
    "build-spec-table": build_spec_table,
    "compute-correlations": compute_correlations,
    "fetch-sequences": fetch_sequences,
    "add-sequences": add_sequences,
    "run-hhblits": run_hhblits,
    "parse-hhr": parse_hhr,
    "filter-hits": filter_hits,
    "fetch-structures": fetch_structures,
    "chain-mapping": chain_mapping,
    "compute-bsasa": compute_bsasa,
    "build-reference": build_reference,
    "build-benchmark": build_benchmark,
}

STEP_ORDER = list(STEPS.keys())


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the requested preprocessing step."""
    parser = argparse.ArgumentParser(
        prog="inm.preprocess",
        description="AP-MS preprocessing pipeline",
    )
    parser.add_argument("step", choices=[*STEPS.keys(), "run-all"])
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)

    if args.step == "run-all":
        for step_name in STEP_ORDER:
            logging.info("Running step: %s", step_name)
            STEPS[step_name].run(cfg, output_dir=args.output_dir)
    else:
        STEPS[args.step].run(cfg, output_dir=args.output_dir)
