import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules.config import SimulationConfig
from modules.data_util import (
    calculate_returns,
    daten_laden,
    get_options_data,
    payoff_vektor,
    real_vola2,
    vek_ret_long,
    vek_ret_short,
)
from modules.optimization import optimize_period

warnings.filterwarnings(
    "ignore", message=".*Incorrect array format.*", category=UserWarning, module="mosek"
)

pd.set_option("display.float_format", "{:.3f}".format)


def run_simulation(config: SimulationConfig | None = None) -> None:
    """Run the monthly bootstrap and portfolio optimization pipeline."""

    if config is None:
        config = SimulationConfig()

    if config.use_seed:
        np.random.seed(config.seed)

    df, df_options, rf = daten_laden(
        Index=config.file_spx,
        Derivate=config.file_options,
        RiskFreeRate=config.file_risk_free,
    )
    options_data = get_options_data(
        df_options, start_index=config.start_index, end_index=config.end_index
    )
    returns_data = calculate_returns(
        df=df,
        start_date_vec=options_data.StartDatumPeriode,
        end_date_vec=options_data.EndDatumPeriode,
        min_periods_z=60,
        risk_adj=config.risk_adjustment,
        rm_lambda=config.rm_lambda,
    )

    K = len(config.d_window)

    rf = rf.dropna()
    rf_t = rf.reindex(options_data.StartDatumPeriode, method="ffill").to_numpy()
    rf_t = rf_t / 12

    S_t_arr = np.empty((options_data.N, K, config.n_wiederholungen), dtype=float)
    for i in range(options_data.N):
        start = options_data.StartDatumPeriode.iloc[i]
        S_0 = float(returns_data.start_prices.loc[start])
        pool = returns_data.stand_log_m.loc[:start].dropna().to_numpy()
        stand_draws = np.random.choice(pool, size=config.n_wiederholungen, replace=True)
        for j, d in enumerate(config.d_window):
            vol_real = float(
                real_vola2(returns_data.log_d_sqr, start, d=d, Skalierung=21)
            )
            S_t_arr[i, j, :] = S_0 * np.exp(stand_draws * vol_real)

    Sim_payoff_array = np.stack(
        [
            payoff_vektor(S_t_arr, np.array(options_data.strikes_atm_put), "p"),
            payoff_vektor(S_t_arr, np.array(options_data.strikes_atm_call), "c"),
            payoff_vektor(S_t_arr, np.array(options_data.strikes_otm_put), "p"),
            payoff_vektor(S_t_arr, np.array(options_data.strikes_otm_call), "c"),
        ],
        axis=-1,
    )
    sim_vek_ret_long = vek_ret_long(Sim_payoff_array, options_data.ask)
    sim_vek_ret_short = vek_ret_short(Sim_payoff_array, options_data.bid)
    sim_ret_array = np.stack([sim_vek_ret_long, sim_vek_ret_short], axis=-1)

    N_perioden = sim_ret_array.shape[0]
    print("Starte parallele Optimierung...")

    results = Parallel(n_jobs=config.n_jobs, verbose=10)(
        delayed(optimize_period)(i, rf_t[i], sim_ret_array[i], config)
        for i in range(N_perioden)
    )

    all_best_weights = []
    vol_scaler = []

    cols = [
        "W_Long_P1",
        "W_Short_P1",
        "W_Long_C1",
        "W_Short_C1",
        "W_Long_P2",
        "W_Short_P2",
        "W_Long_C2",
        "W_Short_C2",
    ]

    for i in range(N_perioden):
        start = options_data.StartDatumPeriode.iloc[i]
        _, best_d_index, best_weights, _ = results[i]

        if best_d_index < 0:
            best_d_value = np.nan
        else:
            best_d_value = config.d_window[int(best_d_index)]
        vol_scaler.append(best_d_value)

        all_best_weights.append(pd.Series(best_weights, index=cols, name=start))
    df_best_w_pro_t = pd.concat(all_best_weights, axis=1).T
    df_best_w_pro_t["Best_d"] = vol_scaler

    log_df = df_best_w_pro_t.copy()
    log_df["RunID"] = config.run_id
    log_df["Gamma"] = config.gamma
    log_df["Seed"] = config.seed if config.use_seed else np.nan
    log_df = log_df.reset_index().rename(columns={"index": "Periode"})

    log_directory = os.path.dirname(config.log_file)
    if log_directory:
        os.makedirs(log_directory, exist_ok=True)
    file_exists = os.path.isfile(config.log_file)
    log_df.to_csv(config.log_file, mode="a", index=False, header=not file_exists)

    print(f"Fertig! Log geschrieben in {config.log_file}")


if __name__ == "__main__":
    run_simulation()
