import numpy as np
import time
from modules.config import SimulationConfig
from main import run_simulation

N_RUNS = 1
START_SEED = 999
GAMMA = 8
N_RUNS2 = 1
d_window_values = [1, 5, 10, 20, 30, 60]


for i in range(N_RUNS):
    print(f"\n=== Starte Global Run {i+1} von {N_RUNS} ===")
    for k in range(N_RUNS2):
        # for val in d_window_values:
        if i == 0:
            bound = 0.02
        if i == 1:
            bound = 0.05
        if i == 2:
            bound = 0.1

        j = i + 2
        run_config = SimulationConfig(
            use_crra=False,
            use_seed=True,
            gamma=10,
            big_array=False,
            n_wiederholungen=1000,
            worst_case=False,
            n_jobs=-8,
            # d_window=[val],
            # eps= 1e-2,
            lB=0.1,
            sB=0.1,
            # max_diff=0.05,
            pair_idx=False,
            # eB= 0.05
            # risk_adjustment="rm",
            # rm_lambda=0.97
        )

        try:
            run_simulation(config=run_config)
        except Exception as e:
            print(f"Fehler in Run {i+1} : {e}")


print("\nAlle Simulationen abgeschlossen.")
