import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename):
    return os.path.join(BASE_DIR, "data", filename)


def get_log_path(filename):
    return os.path.join(BASE_DIR, "results", filename)


@dataclass(frozen=True)
class SimulationConfig:
    file_spx: str = field(default_factory=lambda: get_data_path("SpxDaten.csv"))
    file_options: str = field(
        default_factory=lambda: get_data_path("Databento_Final_Options.csv")
    )
    file_risk_free: str = field(
        default_factory=lambda: get_data_path("DSG1MO_fred.csv")
    )
    log_file: str = field(
        default_factory=lambda: get_log_path("simulation_log_baseline.csv")
    )
    n_wiederholungen: int = 10_000
    use_seed: bool = True
    seed: int = 999
    worst_case: bool = False
    gamma: float = 10.0
    use_crra: bool = True
    big_array: bool = False
    n_assets: int = 8
    n_jobs: int = -1
    eps: float = 1e-3
    start_index: int = 0
    risk_adjustment: str = "rv"
    rm_lambda: float = 0.97
    end_index: Optional[int] = None
    d_window: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 30, 60])
    long_idx: List[int] = field(default_factory=lambda: [0, 2, 4, 6])
    short_idx: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    pair_idx: List[List[int]] = field(
        default_factory=lambda: [[0, 1], [2, 3], [4, 5], [6, 7]]
    )
    bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0)] * 8)
    lB: float = 0.02
    sB: float = 0.02
    eB: float = 1.0
    max_diff: float = 0.05
    w0: Optional[List[float]] = None
    short_only: bool = False
    long_only: bool = False

    @property
    def run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vol_tag = (
            f"rm={self.risk_adjustment}_lambda={self.rm_lambda}"
            if self.risk_adjustment == "rm"
            else f"{self.risk_adjustment}"
        )
        d = f"Vol_scaler{self.d_window}" if len(self.d_window) < 2 else "all"
        return (
            f"{timestamp}_"
            f"gamma={self.gamma}_"
            f"sB={self.sB}_"
            f"lB={self.lB}_"
            f"Max_diff={self.max_diff}_"
            f"VOL={d}_"
            f"VolMeth={vol_tag}"
        )
