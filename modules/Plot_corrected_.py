import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm


def load_and_prep_data(
    filepath, filepath_ret, filepath_rates, filepath_spx, filepath_options=None
):
    df = pd.read_csv(filepath)
    df_ret = pd.read_csv(filepath_ret)
    df_rates = pd.read_csv(filepath_rates)
    sp500_df = pd.read_csv(filepath_spx)

    df["Periode"] = pd.to_datetime(df["Periode"])
    df_ret["Periode"] = pd.to_datetime(df_ret["Periode"], dayfirst=False)
    df_rates["Date"] = pd.to_datetime(df_rates["Date"])
    df = df.sort_values("Periode")
    df_rates = df_rates.sort_values("Date")

    sp500_df = pd.read_csv(filepath_spx)
    sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])

    df = pd.merge_asof(
        df, df_rates, left_on="Periode", right_on="Date", direction="backward"
    )
    df.rename(columns={"DGS1MO": "RiskFreeRate"}, inplace=True)

    weight_cols = [c for c in df.columns if c.startswith("W_")]

    df["Total_Exposure"] = df[weight_cols].sum(axis=1)

    long_cols = [c for c in weight_cols if "Long" in c]
    short_cols = [c for c in weight_cols if "Short" in c]

    df["Long_Exposure"] = df[long_cols].sum(axis=1)
    df["Short_Exposure"] = df[short_cols].sum(axis=1)
    df["Cash_Weight"] = (1.0 - df["Long_Exposure"] + df["Short_Exposure"]).clip(
        lower=0.0
    )

    return df, df_ret, weight_cols, long_cols, short_cols, sp500_df


def calculate_performance_metrics(strategy_returns, rf_returns, gamma=4):
    excess_returns = strategy_returns - rf_returns

    ann_factor = 12
    mean_ann = strategy_returns.mean() * ann_factor
    std_ann = strategy_returns.std() * np.sqrt(ann_factor)

    sharpe_ratio = (excess_returns.mean() * ann_factor) / (
        excess_returns.std() * np.sqrt(ann_factor)
    )
    skew = stats.skew(strategy_returns)
    exc_kurt = stats.kurtosis(strategy_returns)

    wealth = 1 + strategy_returns

    if gamma == 1:
        utility = np.log(wealth)
        ce_ann = (np.exp(utility.mean()) ** ann_factor) - 1
    else:
        utility = (wealth ** (1 - gamma)) / (1 - gamma)
        mean_utility = utility.mean()
        ce_monthly = ((1 - gamma) * mean_utility) ** (1 / (1 - gamma)) - 1
        ce_ann = (1 + ce_monthly) ** ann_factor - 1

    return {
        "Ann. Mean": mean_ann,
        "Ann. Std": std_ann,
        "Skewness": skew,
        "Excess Kurtosis": exc_kurt,
        "Ann. Sharpe Ratio": sharpe_ratio,
        "Ann. CE (gamma=4)": ce_ann,
    }
