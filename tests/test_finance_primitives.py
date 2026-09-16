import numpy as np

from modules.data_util import option_payoff, payoff_vektor, vek_ret_long, vek_ret_short
from modules.optimization import crra_utility


def test_call_and_put_payoffs() -> None:
    spot = np.array([80.0, 100.0, 120.0])
    np.testing.assert_array_equal(option_payoff(spot, 100.0, "c"), [0.0, 0.0, 20.0])
    np.testing.assert_array_equal(option_payoff(spot, 100.0, "p"), [20.0, 0.0, 0.0])


def test_vectorized_payoffs_broadcast_strikes_by_period() -> None:
    spot = np.array([[[90.0, 110.0]], [[180.0, 220.0]]])
    strikes = np.array([100.0, 200.0])
    result = payoff_vektor(spot, strikes, "c")
    np.testing.assert_array_equal(result, [[[0.0, 10.0]], [[0.0, 20.0]]])


def test_long_and_short_security_returns_include_entry_price() -> None:
    payoff = np.array([0.0, 15.0])
    np.testing.assert_allclose(vek_ret_long(payoff, 10.0), [-1.0, 0.5])
    np.testing.assert_allclose(vek_ret_short(payoff, 10.0), [1.0, -0.5])


def test_crra_utility_matches_closed_form() -> None:
    returns = np.array([0.0, 0.1])
    expected = np.mean((1 + returns) ** -3 / -3)
    assert np.isclose(crra_utility(returns, gamma=4), expected)
