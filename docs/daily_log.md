# Daily Log

## Day 2 — Tuesday
**What I did:** Downloaded 5 years of USD/INR daily OHLCV data using yfinance 
(ticker: INR=X), saved raw CSV to data/raw/USDINR_daily_raw.csv.

**What I noticed:** Dataset has ~1300 rows (trading days only, no weekends). 
The Close column is the INR price per 1 USD (e.g. ~83.5).

**One question I have:** Why does yfinance sometimes return fewer rows than 
expected for "5y" — are holidays and exchange closures excluded automatically?


## Day 3 — Wednesday
**What I did:** Downloaded 5 years of USD/BRL daily data (ticker: BRL=X), 
merged with USD/INR on the Date column, derived synthetic INR/BRL cross rate 
using formula: USDINR_Close / USDBRL_Close. Saved 3 CSV files total.

**What I noticed:** Merged dataset has ~1300 rows. Mean INR/BRL ≈ 15.4738, which is 
economically reasonable.

**One question I have:** The inner join dropped some rows — should I use a 
forward-fill on missing dates instead, to avoid losing data points?

### 🧠 Why this formula works
```
1 USD = 83.5 INR
1 USD = 5.0  BRL
─────────────────
∴ 1 BRL = 83.5 / 5.0 = 16.7 INR


## Day 4 — Thursday
**What I did:** Downloaded hourly USD/INR and USD/BRL data (2y, USDINR -- > 10549 and USDBRL --> 8416). 
Fetched India long-term interest rate from FRED via pandas_datareader. 
Printed full data inventory table for all 5 datasets.

**What I noticed:** Hourly data has many more rows than daily. 
FRED series is monthly — much lower frequency than price data. Hourly data 
has some null rows around weekends/market close hours.

Dataset File          Rows  
    USDINR daily      1302     
    USDBRL daily      1302     
INRBRL synthetic      1300    
   USDINR hourly     10549     
   USDBRL hourly      8416     
 India Repo Rate        85     

**One question I have:** When I later merge hourly price data with monthly 
macro data, how do I forward-fill monthly values across daily/hourly rows 
without introducing look-ahead bias?



## Day 6 — Saturday
**What I did:** Created 02_data_cleaning.ipynb. Standardized column names, 
parsed dates as index, forward-filled gaps ≤ 2 days, dropped gaps > 3 days, 
flagged outliers using 30-day rolling mean ± 3 std. Saved 3 clean CSVs to 
data/processed/.

**What I noticed:** Most dropped rows were from long holiday periods. Outlier 
flags were rare (<1% of rows) — mostly during COVID volatility period (2020). 
The is_outlier column lets me audit them later without losing data.

**One question I have:** Should the rolling window for outlier detection be 
the same (30 days) for both daily and hourly data, or should it be scaled 
by frequency?



## Day 7 — Sunday
**What I did:** Built 4 thesis-ready EDA plots — price history, returns 
distribution, rolling volatility, and correlation heatmap. Wrote markdown 
observations after each plot. Confirmed heavy tails in returns (kurtosis > 0, 
normality test p < 0.05) which is the key statistical justification for SNNs.

**What I noticed:** The Q-Q plot clearly shows fat tails. Volatility spikes 
align precisely with known macro events. INR/BRL correlation structure differs 
from its USD components — good sign it carries independent signal.

**One question I have:** For the SNN model, should I use raw returns or 
volatility-normalised returns (e.g., divide by rolling std) as the input 
signal to the spike encoder?


## Day 11 — Thursday
**What I did:** Installed SpikingJelly and PyTorch. Created a single LIF 
neuron with tau=2.0, v_threshold=1.0. Fed a 5-step input sequence and 
traced membrane voltage step by step. Tested 4 parameter combinations 
(tau × v_threshold) and plotted all comparisons.

**What I noticed:** The neuron resets to 0 after firing, so a spike 
"spends" the accumulated voltage. With tau=5.0 the voltage builds faster 
because old inputs decay slower — counter-intuitive at first but makes 
sense once you see the trace. v_threshold has a more direct effect on 
spike count than tau does.

**One question I have:** When I chain multiple LIF layers into a full 
network, does each layer need its own tau, or do all layers share the 
same parameter? Can tau be learned during training?
```

---

### 🧠 The parameter map for your thesis
```
SNN Parameter    FX Meaning
─────────────────────────────────────────────────────
tau = 2.0        Recent price moves weighted more than old ones
v_threshold      Minimum accumulated signal to trigger settlement alert  
v_reset = 0.0    After a settlement event fires, system resets cleanly
spike = 1        "This is a significant cross-rate movement"
spike = 0        "Normal day, no settlement action needed"

