"""Recreate the public performance figures from the selected allocation log."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = ROOT / "log" / "simulation_log_main.csv"
ASSETS = ROOT / "assets"

WEIGHT_COLUMNS = [
    "W_Long_P1",
    "W_Short_P1",
    "W_Long_C1",
    "W_Short_C1",
    "W_Long_P2",
    "W_Short_P2",
    "W_Long_C2",
    "W_Short_C2",
]


def load_baseline_returns() -> tuple[pd.Series, pd.Series]:
    log = pd.read_csv(LOG, parse_dates=["Periode"])
    baseline_mask = (
        log["RunID"].str.contains("sB=0.02_lB=0.02_Max_diff=0.05", regex=False)
        & log["Gamma"].eq(10)
        & log["Seed"].eq(999)
    )
    baseline = log.loc[baseline_mask].copy()
    if baseline.empty:
        raise RuntimeError("The primary-specification allocation log was not found.")

    run_id = baseline["RunID"].iloc[0]
    weights = (
        baseline.loc[baseline["RunID"].eq(run_id)].set_index("Periode").sort_index()
    )

    realized = (
        pd.read_csv(DATA / "RR_DF_FINAL.csv", parse_dates=["Periode"])
        .set_index("Periode")
        .sort_index()
    )
    rates = (
        pd.read_csv(DATA / "DSG1MO_fred.csv", parse_dates=["Date"])
        .set_index("Date")
        .sort_index()["DGS1MO"]
    )

    long_columns = [column for column in WEIGHT_COLUMNS if "Long" in column]
    short_columns = [column for column in WEIGHT_COLUMNS if "Short" in column]
    option_return = (weights[WEIGHT_COLUMNS] * realized[WEIGHT_COLUMNS]).sum(axis=1)
    cash_weight = (
        1.0 - weights[long_columns].sum(axis=1) + weights[short_columns].sum(axis=1)
    )
    monthly_rate = rates.reindex(weights.index, method="ffill") / 12
    oops_return = option_return + cash_weight * monthly_rate

    spx = (
        pd.read_csv(DATA / "SpxDaten.csv", parse_dates=["Date"])
        .set_index("Date")
        .sort_index()["Close"]
    )
    spx_at_rebalancing = spx.reindex(weights.index, method="ffill")
    spx_return = spx_at_rebalancing.pct_change().dropna()
    return oops_return.rename("OOPS"), spx_return.rename("S&P 500")


def save_performance_figures(oops_return: pd.Series, spx_return: pd.Series) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # Align completed OOPS holding periods with the next rebalancing dates.
    aligned_oops = oops_return.iloc[:-1].copy()
    aligned_oops.index = spx_return.index
    wealth = pd.DataFrame(
        {
            "OOPS": (1 + aligned_oops).cumprod(),
            "S&P 500": (1 + spx_return).cumprod(),
        }
    )
    initial = pd.DataFrame(
        {"OOPS": [1.0], "S&P 500": [1.0]}, index=[oops_return.index[0]]
    )
    wealth = pd.concat([initial, wealth])

    colors = {"OOPS": "#0B5CAD", "S&P 500": "#747B84"}
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for column in wealth:
        ax.plot(
            wealth.index,
            wealth[column],
            label=column,
            color=colors[column],
            linewidth=2.2,
        )
    ax.set_title("Cumulative wealth of $1")
    ax.set_ylabel("Portfolio value")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "cumulative_wealth.png", dpi=180)
    plt.close(fig)

    drawdown = wealth.div(wealth.cummax()).sub(1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for column in drawdown:
        ax.plot(
            drawdown.index,
            drawdown[column],
            label=column,
            color=colors[column],
            linewidth=1.8,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "drawdowns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(
        oops_return,
        bins=25,
        color=colors["OOPS"],
        alpha=0.88,
        edgecolor="white",
    )
    ax.axvline(
        oops_return.mean(),
        color="#C33C54",
        linestyle="--",
        linewidth=1.6,
        label="Mean",
    )
    ax.set_title("Distribution of monthly OOPS returns")
    ax.set_xlabel("Monthly return")
    ax.set_ylabel("Observations")
    ax.xaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "monthly_return_distribution.png", dpi=180)
    plt.close(fig)


def print_summary(oops_return: pd.Series, spx_return: pd.Series) -> None:
    for returns in (oops_return, spx_return):
        annual_mean = returns.mean() * 12
        annual_volatility = returns.std() * np.sqrt(12)
        wealth = (1 + returns).cumprod()
        maximum_drawdown = wealth.div(wealth.cummax()).sub(1).min()
        print(
            f"{returns.name}: annual mean={annual_mean:.2%}, "
            f"annual volatility={annual_volatility:.2%}, "
            f"maximum drawdown={maximum_drawdown:.2%}"
        )


def main() -> None:
    oops_return, spx_return = load_baseline_returns()
    save_performance_figures(oops_return, spx_return)
    print_summary(oops_return, spx_return)


if __name__ == "__main__":
    main()
