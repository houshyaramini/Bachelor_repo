import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from modules.Plot_corrected_ import (
    load_and_prep_data,
     calculate_performance_metrics
)
from modules.optimization import crra_utility

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 500)
filepath = "log/simulation_log6.csv"
filepath_returns = "data/RR_DF_FINAL.csv"
filepath_rates = "data/DSG1MO_fred.csv"
filepath_spx = "data/SpxDaten.csv"
filepath_vix = "data/VIX.csv"

df_vix = pd.read_csv(filepath_vix, parse_dates=["Date"], index_col="Date")

df, df_returns, w_cols, l_cols, s_cols, sp500_df = load_and_prep_data(
    filepath, filepath_returns, filepath_rates, filepath_spx
)


unique_runs = sorted(df["RunID"].unique())
run = -3
test = df[df["RunID"] == unique_runs[run]].copy()


test["ATM_Put"] = test.loc[:, "W_Long_P1"] - test.loc[:, "W_Short_P1"]
test["ATM_Call"] = test.loc[:, "W_Long_C1"] - test.loc[:, "W_Short_C1"]
test["OTM_Put"] = test.loc[:, "W_Long_P2"] - test.loc[:, "W_Short_P2"]
test["OTM_Call"] = test.loc[:, "W_Long_C2"] - test.loc[:, "W_Short_C2"]


Assets = test[["ATM_Put","ATM_Call","OTM_Put","OTM_Call","Cash_Weight"]]
print(Assets.describe())
print(test['Best_d'].value_counts())
#print(Assets)

def auswertung(last_run: int = -1):

    df_vix = pd.read_csv(filepath_vix, parse_dates=["Date"], index_col="Date")
    test = df[df["RunID"] == unique_runs[last_run]].copy()
    test.loc[:, "RiskFreeRate"] = test.loc[:, "RiskFreeRate"] / 12
    test["zinsen"] = test["Cash_Weight"] * test["RiskFreeRate"]
    test = test.set_index("Periode")
    test_series = test.index.to_series()

    N = len(test_series)
    Start_vec = test_series[:N]
    Mat_vix = df_vix["Close"].reindex(Start_vec, method="ffill")
    w_df = test.filter(like="W_")
    ret_df = df_returns.set_index("Periode").filter(like="W_")
    port_df = ret_df * w_df
    port_df["zinsen"] = test["zinsen"]
    port_df["ATM_Put"] = port_df["W_Long_P1"] + port_df["W_Short_P1"]
    port_df["ATM_Call"] = port_df["W_Long_C1"] + port_df["W_Short_C1"]
    port_df["OTM_Put"] = port_df["W_Long_P2"] + port_df["W_Short_P2"]
    port_df["OTM_Call"] = port_df["W_Long_C2"] + port_df["W_Short_C2"]
    port_df["Excess_ret"] = port_df[w_cols].sum(axis=1)
    port_df["TotalRet"] = port_df["zinsen"] + port_df["Excess_ret"]
    port_df["Wealth"] = 100 * (port_df["TotalRet"] + 1).cumprod(axis=0)
    port_df["ExcessWealth"] = 100 * (port_df["Excess_ret"] + 1).cumprod(axis=0)
    port_df["RiskFreeRate"] = test["RiskFreeRate"]
    op = port_df.iloc[:, -9:]
    op["VIX"] = Mat_vix
    op["Peak"] = op["Wealth"].cummax()
    op["Drawdown"] = (op["Wealth"] - op["Peak"]) / op["Peak"]

    return op

tt = auswertung(run)
tt = tt.dropna()
print(tt)
metrics = calculate_performance_metrics(tt["TotalRet"], tt["RiskFreeRate"], gamma=4)
print("--- Performance Metriken ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")






