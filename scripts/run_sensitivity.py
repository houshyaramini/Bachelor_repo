"""Run the 8 x 8 position-cap and net-short-limit sensitivity grid."""

import argparse
from pathlib import Path

from main import run_simulation
from modules.config import SimulationConfig

POSITION_CAPS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20)
NET_SHORT_LIMITS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/simulation_log_sensitivity.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()

    for position_cap in POSITION_CAPS:
        for net_short_limit in NET_SHORT_LIMITS:
            print(
                "Running sensitivity specification: "
                f"position_cap={position_cap:.0%}, "
                f"net_short_limit={net_short_limit:.0%}"
            )
            config = SimulationConfig(
                n_wiederholungen=args.scenarios,
                use_seed=True,
                seed=999,
                gamma=10.0,
                use_crra=True,
                n_jobs=args.jobs,
                eps=1e-3,
                risk_adjustment="rv",
                d_window=[1, 5, 10, 20, 30, 60],
                lB=position_cap,
                sB=position_cap,
                eB=1.0,
                max_diff=net_short_limit,
                log_file=str(output),
            )
            run_simulation(config)


if __name__ == "__main__":
    main()
