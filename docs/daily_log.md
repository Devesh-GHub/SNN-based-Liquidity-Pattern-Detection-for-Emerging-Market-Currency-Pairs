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


## Day 13 — Saturday
**What I did:** Researched real SBI and HDFC outward remittance fees, 
SWIFT correspondent bank charges, and Brazil IOF tax. Built a complete 
cost formula for a ₹10,00,000 INR→BRL transaction. Calculated current 
route total at ₹42,080 (4.21%) vs proposed at ₹1,180 (0.12%). Wrote 
the User Story for a Surat textile exporter. Saved as docs/fee_research.md.

**What I noticed:** The biggest cost is NOT the flat bank fee (₹1,000) —  
it's the hidden FX spread (₹35,000+). This is the academically important 
insight: the markup is invisible to the exporter until they compare 
against mid-market rates. My SNN system's value proposition is eliminating 
the two-leg conversion, not just the flat fee.

**One question I have:** The 0.1% proposed fee assumption will be 
challenged in any thesis review. Should I cite a specific BIS or RBI 
paper on DLT/blockchain settlement costs to back it up?


## Day 14 — Sunday
**What I did:** Read RBI CBDC Concept Note (Executive Summary + Wholesale 
section) and BIS mBridge project documentation. Extracted structured policy 
notes with direct quotes. Wrote one-page project_motivation.md covering 
problem, why solutions fail, why SNN, policy alignment, and User Story.

**What I noticed:** Neither the RBI document nor mBridge mentions AI or ML 
anywhere. This is the gap — infrastructure is being built but the 
intelligence layer for optimal settlement timing doesn't exist yet. That's 
exactly what our project contributes.

**One question I have:** mBridge involved RBI only as an "observer member" 
— India has not committed to it. Should I position my project as supporting 
India's *own* bilateral INR/BRL corridor instead of as a BRICS-wide system, 
to make the policy case stronger and more specific?


## Day 15 — Monday
**What I did:** Built 9 features from INR/BRL daily data: 5 price-based 
(return, log return, rolling mean/std, momentum), 3 spike-based (signal, 
intensity, inter-spike interval), 1 macro (repo rate). Created binary 
direction target. Ran leakage audit. Saved feature_matrix_daily.csv. 
Printed correlation table.

**What I noticed:** Correlations with target are all below 0.05 — typical 
for FX data. The inter-spike interval feature is unique — no standard 
finance ML paper uses it. Low linear correlation doesn't mean the SNN 
can't learn from it: SNNs capture temporal non-linear patterns that 
Pearson correlation completely misses.

**One question I have:** Should I add a "time of week" feature (day 
of week as integer 0-4)? Monday and Friday often show different FX 
behaviour due to position squaring. Or does that risk overfitting on 
a 5-year window?
```

---

### 🧠 Why each feature earns its place
```
Feature                  Why it belongs
─────────────────────────────────────────────────────────────────
daily_return             Raw signal — the fundamental input
log_return               Stabilised version — better for gradient-based training
rolling_mean_7d          Trend context — is the rate rising or falling?
rolling_std_7d           Volatility context — is now a calm or stressed period?
price_momentum_5d        Medium-term direction — week-scale trend
spike_signal             Event flag — did anything significant happen today?
spike_intensity          Event magnitude — how significant was it?
inter_spike_interval     Calm duration — how long since last stress event?  ← unique
india_repo_rate          Macro anchor — monetary policy context
─────────────────────────────────────────────────────────────────
target                   Tomorrow's direction — what we predict



## Day 17 — Wednesday
**What I did:** Created 07_lstm_baseline_prep.ipynb. Wrote a 5-sentence 
LSTM explanation in my own words. Built create_sequences() function — 
converts flat feature matrix into (samples, 10, 9) sliding windows. 
Applied to train and val sets. Manually verified sample 0 has no future 
data leakage. Wrote LSTM architecture plan in markdown.

**What I noticed:** After the lookback window, I lose 10 rows from each 
split — small but expected. The sequence shape (samples, 10, 9) makes 
the data leakage check visual and obvious: X is always rows t-10 to t-1, 
y is always row t. The LSTM treats all 10 days equally; the SNN will 
weight recent spikes more via the LIF membrane decay.

**One question I have:** Should I normalise features BEFORE creating 
sequences (normalise → sequence) or AFTER (sequence → normalise)? And 
should normalisation parameters (mean, std) be fitted on train only, 
then applied to val and test?


## Day 18 — Thursday
**What I did:** Built hourly feature matrix from USDINR/USDBRL hourly 
clean data. Derived hourly INR/BRL cross rate. Calibrated spike threshold 
to ~0.1% (hourly moves are ~5x smaller than daily). Added time-of-day 
feature (0-23 UTC) — spike rate varies meaningfully by session. Saved 
feature_matrix_hourly.csv. Compared daily vs hourly spike consistency.

**What I noticed:** Spike rate is higher in European session overlap 
(07-13 UTC) — this makes sense as London is the dominant FX centre and 
INR/BRL both react to European risk sentiment. The consistency check 
confirms that days with daily spikes also tend to have multiple hourly 
spikes — the two scales are telling the same story at different resolution.

**One question I have:** For the dashboard demo, should I show the 
hourly spike train as a live-updating chart? If yes, I need to think 
about how to simulate "streaming" data from the hourly CSV.
```

---

### 🧠 Daily vs hourly — the two-scale picture
```
Scale     Threshold   Captures                  Used for
────────────────────────────────────────────────────────────
Daily     0.5%        Day-level regime shifts   SNN model training
                      (COVID, policy decisions)  thesis evaluation

Hourly    0.1%        Session-level volatility  Dashboard live feed
                      (London open, NY overlap)  real-time demo feel


## Day 19 — Friday
**What I did:** Updated README.md with full project state — problem, 
solution, data, folder structure, environment setup, progress checklist, 
key research numbers, and customer profile. Ran version collector to get 
exact package versions. Git committed all Week 3 work.

**What I noticed:** Writing the README forced me to see the whole project 
at once. Month 1 is more complete than it feels day-to-day — 14 checkboxes 
ticked. The README is also the document I would show a professor or 
supervisor tomorrow and it would make sense to them.

**One question I have:** For Month 2, should I add a requirements.txt 
file (pip freeze > requirements.txt) in addition to the README install 
instructions, so the environment is fully reproducible without manually 
specifying versions?

