# Data dictionary and provenance

This directory contains the compact inputs used by the thesis code. It does
not contain the original tick-level OPRA download.

| File | Contents | Source |
|---|---|---|
| `Databento_Final_Options.csv` | Monthly selected SPX option bid/ask quotes, strikes, and maturities | Databento OPRA.PILLAR |
| `SpxDaten.csv` | Daily S&P 500 index levels | Yahoo Finance / `yfinance` |
| `VIX.csv` | Daily VIX history used in the empirical analysis | Yahoo Finance / `yfinance` |
| `DSG1MO_fred.csv` | One-month Treasury yield in decimal form | FRED series DGS1MO |
| `RR_DF_FINAL.csv` | Realized returns for the eight long/short option securities | Author calculation |
| `option_greeks.csv` | Derived option characteristics and Black-Scholes Greeks | Author calculation |

## Important licensing notice

The source providers and exchanges retain any rights they hold in their market
data. Inclusion of a compact research input in this repository does not grant
downstream users a license to redistribute or commercially use the underlying
data. Users are responsible for reviewing the applicable provider and exchange
terms and, where required, obtaining the data directly.

The code and derived methodology can be inspected independently of the source
data. If public redistribution of a particular input is not permitted, replace
it with a locally obtained file using the same schema described above.
