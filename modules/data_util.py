import pandas as pd
import numpy as np
from arch import arch_model
from dataclasses import dataclass
from typing import Union, Optional


def daten_laden(Index, Derivate, RiskFreeRate):
    df = pd.read_csv(Index, parse_dates=["Date"], index_col="Date")
    df_options = pd.read_csv(
        Derivate, parse_dates=["BewertungsDatum"], index_col="BewertungsDatum"
    )
    rf = pd.read_csv(RiskFreeRate, parse_dates=["Date"], index_col="Date")
    return df, df_options, rf


def real_vola2(ret, start, d, Skalierung):
    werte = ret[:start][-d:]
    return np.sqrt(werte.sum()) * np.sqrt(Skalierung / d)


def option_payoff(Spot, Strike, type):
    if type == "c":
        return np.maximum(Spot - Strike, 0)
    if type == "p":
        return np.maximum(Strike - Spot, 0)


def payoff_vektor(s_t, k, options_type):
    if s_t.ndim == 3:
        k = np.array(k)
        k_array = k[:, np.newaxis, np.newaxis]
        if options_type == "c":
            return np.maximum(s_t - k_array, 0)
        if options_type == "p":
            return np.maximum(k_array - s_t, 0)
    else:
        return option_payoff(Spot=s_t, Strike=k, type=options_type)


def vek_ret_long(payoff, ask):
    if payoff.ndim == 4:
        ask_array = ask[:, np.newaxis, np.newaxis, :]
    else:
        ask_array = ask
    return (payoff / ask_array) - 1


def vek_ret_short(payoff, bid):
    if payoff.ndim == 4:
        bid_array = bid[:, np.newaxis, np.newaxis, :]
    else:
        bid_array = bid
    return 1 - (payoff / bid_array)


@dataclass
class OptionsData:
    ask: np.ndarray
    bid: np.ndarray
    strikes_all: np.ndarray
    strikes_atm_call: np.ndarray
    strikes_atm_put: np.ndarray
    strikes_otm_call: np.ndarray
    strikes_otm_put: np.ndarray
    maturity: pd.Series
    StartDatumPeriode: pd.Series
    EndDatumPeriode: pd.Series
    N: int


def get_options_data(
    df_options: pd.DataFrame, start_index: int = 0, end_index: Optional[int] = None
) -> OptionsData:
    df_options = df_options.iloc[start_index:end_index]
    df_clean = df_options.dropna()
    df_strikes = df_clean.filter(like="Strike")
    maturity = df_options.index.to_series().dropna()
    N = len(maturity)
    return OptionsData(
        ask=df_clean.filter(like="Ask").to_numpy(),
        bid=df_clean.filter(like="Bid").to_numpy(),
        strikes_all=df_strikes.to_numpy(),
        strikes_atm_call=df_strikes.filter(like="ATM_Strike_Call").squeeze().to_numpy(),
        strikes_atm_put=df_strikes.filter(like="ATM_Strike_Put").squeeze().to_numpy(),
        strikes_otm_call=df_strikes.filter(like="OTM_Call_Strike").squeeze().to_numpy(),
        strikes_otm_put=df_strikes.filter(like="OTM_Put_Strike").squeeze().to_numpy(),
        maturity=maturity,
        StartDatumPeriode=maturity,
        EndDatumPeriode=df_clean["Maturity"].squeeze(),
        N=N,
    )


@dataclass
class ReturnsData:

    log_d: pd.Series
    log_d_sqr: pd.Series
    log_m: pd.Series
    rv_m: pd.Series
    stand_log_m: pd.Series
    z_score: pd.Series
    start_prices: Union[pd.Series, float]
    end_prices: np.ndarray


def calculate_returns(
    df: pd.DataFrame,
    start_date_vec: pd.Series,
    end_date_vec: pd.Series,
    min_periods_z: int = 60,
    risk_adj: str = "rv",
    rm_lambda: float = 0.97,
) -> ReturnsData:

    log_d = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    log_d_sqr = log_d**2
    squared_sum = log_d_sqr.resample("ME").sum()
    log_m = log_d.resample("ME").sum()
    rv_m = np.sqrt(squared_sum)
    rv_m_prev = rv_m.shift(1)
    if risk_adj == "rv":
        stand_log_m = log_m / rv_m_prev
    if risk_adj == "rm":
        daily_var = log_d_sqr.ewm(alpha=1 - rm_lambda, adjust=False).mean()
        rm_vol = np.sqrt(daily_var.resample("ME").last() * 21)
        stand_log_m = log_m / rm_vol.shift(1)
    if risk_adj == "ewma":
        ###EWMA
        rv_ewma = rv_m.ewm(span=3).mean()
        stand_log_m = log_m / rv_ewma.shift(1)
    if risk_adj == "GARCH":
        log_d_scaled = log_d * 100
        model = arch_model(log_d_scaled, vol="GARCH", p=1, q=1)
        res = model.fit(disp="off")
        cond_vol_daily = res.conditional_volatility / 100
        cond_vol_monthly = cond_vol_daily.resample("ME").last() * np.sqrt(21)
        stand_log_m = log_m / cond_vol_monthly.shift(1)

    expanding_mean = log_m.expanding(min_periods=min_periods_z).mean().shift(1)
    z = (log_m - expanding_mean) / rv_m.shift(1)
    start_prices = df["Close"].reindex(start_date_vec, method="ffill").squeeze()
    end_dates_dt = pd.to_datetime(end_date_vec)
    days_to_shift = np.where(
        end_dates_dt.dt.dayofweek == 5,
        2,
        np.where(
            end_dates_dt.dt.dayofweek == 4,
            1,
            np.where(end_dates_dt.dt.dayofweek == 3, 1, 0),
        ),
    )
    lookup_dates = end_dates_dt - pd.to_timedelta(days_to_shift, unit="D")
    end_prices = df["Close"].reindex(lookup_dates, method="ffill").to_numpy()

    return ReturnsData(
        log_d=log_d,
        log_d_sqr=log_d_sqr,
        log_m=log_m,
        rv_m=rv_m,
        stand_log_m=stand_log_m,
        z_score=z,
        start_prices=start_prices,
        end_prices=end_prices,
    )
