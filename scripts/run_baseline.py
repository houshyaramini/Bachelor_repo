"""Run the primary thesis specification.

The parameters in this file correspond to the specification reported in the
abstract and Table 4 of the thesis.
"""

from main import run_simulation
from modules.config import SimulationConfig, get_log_path


def main() -> None:
    config = SimulationConfig(
        n_wiederholungen=10_000,
        use_seed=True,
        seed=999,
        gamma=10.0,
        use_crra=True,
        n_jobs=-1,
        eps=1e-3,
        risk_adjustment="rv",
        d_window=[1, 5, 10, 20, 30, 60],
        lB=0.02,
        sB=0.02,
        eB=1.0,
        max_diff=0.05,
        log_file=get_log_path("simulation_log_baseline.csv"),
    )
    run_simulation(config)


if __name__ == "__main__":
    main()
