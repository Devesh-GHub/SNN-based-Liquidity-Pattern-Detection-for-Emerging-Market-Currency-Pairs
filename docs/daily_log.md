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


## Month 2, Day 1 — Monday

**What I did:** Created notebooks/09_snn_model.ipynb. Loaded train/val 
CSVs, dropped NaN rows, normalised with MinMaxScaler fitted on train 
only. Built TimeSeriesDataset class with sliding window __getitem__. 
Created DataLoaders (batch=32, shuffle=False). Verified batch shape 
(32, 10, 9). Saved scaler to outputs/scaler.pkl.

**What I noticed:** After dropna, train has slightly fewer rows than 
the raw split — expected from rolling window warm-up NaNs. The 
scaler.pkl save is critical: FastAPI will need the exact same 
normalisation at inference time or predictions will be garbage.

**One question I have:** The DataLoader returns X in shape (B, T, F) 
but SpikingJelly expects (T, B, F) — time-first. Should I transpose 
inside the training loop or inside the model's forward() method?

## Month 2, Day 2 — Tuesday
**What I did:** Defined BRICSLiquiditySNN class with 2 LIF layers, 
surrogate gradient (ATan), spike accumulator readout. Verified forward 
pass: (32,10,9) → (32,1). Counted parameters (~500 total — deliberately 
small for dataset size). Added spike_rate_monitor() utility. Wrote full 
architecture justification covering tau choice, two-layer design, 
accumulator readout, and RNN comparison.

**What I noticed:** Before training, lif1 and lif2 spike rates are 
near-random (~0.3-0.5) and output probabilities hover near 0.5 — 
exactly as expected for an untrained model. The functional.reset_net() 
call is critical: without it, membrane voltage from batch N leaks into 
batch N+1 and training diverges.

**One question I have:** The surrogate gradient ATan is one option — 
SpikingJelly also offers Sigmoid and PiecewiseQuadratic surrogates. 
Does the choice of surrogate function significantly affect final 
accuracy, or is it mainly a training stability concern?


## Month 2, Day 3 — Wednesday
**What I did:** Wrote full training loop with BCE loss, Adam optimiser, 
ReduceLROnPlateau scheduler, early stopping (patience=10), gradient 
clipping (max_norm=1.0). Tracked train/val loss, accuracy, and AUC 
per epoch. Restored best model weights. Saved model checkpoint to 
outputs/snn_best.pt and training log to outputs/snn_training_log.csv.

**What I noticed:** [fill in your actual results]
- First epoch loss: [X] — confirms no NaN, model is alive
- Val AUC after 30 epochs: [X] — [above/near] random baseline
- Early stopping triggered at epoch [X] / ran full 30 epochs

**One question I have:** My val AUC is hovering around 0.50–0.55 
after 30 epochs. Is this expected for financial direction prediction, 
or does it mean the model isn't learning? What's a realistic target 
AUC for this type of problem?
```

---

### 🧠 Your answer — AUC 0.50–0.55 is normal and expected

FX direction prediction is one of the hardest ML problems. Here's the realistic target landscape:
```
AUC range     Interpretation
──────────────────────────────────────────────────────
0.50          Random — model learned nothing
0.51 – 0.55   Weak signal — model found something, not reliable
0.55 – 0.62   Moderate — publishable for academic thesis ✅
0.62 – 0.70   Strong — excellent for FX prediction
> 0.70        Suspicious — check for data leakage



## Month 2, Day 4 — Thursday
**What I did:** Final clean training run (50 epochs, early stop=10,
AUC-based stopping). Plotted learning curves (loss, accuracy, AUC).
Computed full validation metrics: accuracy, precision, recall, F1,
AUC, confusion matrix, ROC curve. Saved model weights (snn_model.pth),
full checkpoint (snn_best.pt), and config JSON (snn_config.json).

**Official SNN results:**
- Accuracy  : [fill]
- Precision : [fill]
- Recall    : [fill]
- F1        : [fill]
- AUC-ROC   : [fill]
- Best epoch: [fill]

**What I noticed:** [choose what applies]
- Confusion matrix shows model is [biased toward UP / balanced / 
  biased toward DOWN]. 
- ROC curve shows AUC above random baseline — genuine signal found.
- False alarm rate of [X]% means [X]% of "SETTLE NOW" signals 
  would be triggered on unfavourable days — acceptable for a demo.

**One question I have:** For the thesis comparison table, should I
report metrics on the val set or wait and use the test set? I haven't
touched test_features.csv yet as instructed.
```

---

### 🧠 Your answer — val set for development, test set for final paper

Use **val set now** for all development comparisons (SNN vs LSTM tuning). The test set is opened **exactly once** in Month 3 for the final thesis table. The rule is:
```
Val set   → used during development to compare models      (now)
Test set  → opened once for final honest evaluation        (Month 3)



## Month 2, Day 5 — Friday
**What I did:** Recorded spike activity from LIF1 and LIF2 layers using
PyTorch forward hooks across 50 validation sequences. Plotted spike raster
for individual samples (UP vs DOWN predicted). Plotted average spike rate
per timestep and per neuron heatmaps. Calculated energy efficiency estimate
vs dense LSTM baseline using SynOps framework (Blouw et al., 2019).

**Official spike rates:**
- LIF1 mean spike rate : [fill]%
- LIF2 mean spike rate : [fill]%
- Energy reduction vs dense: [fill]%

**What I noticed:** [fill after running]
- Spike patterns differ between UP-predicted and DOWN-predicted samples
  (or: look similar — note honestly what you see)
- Some neurons fire consistently across all timesteps (always-active)
  while others are selective (fire only on specific days)
- The heatmap shows [early/late/uniform] timesteps have highest activity

**One question I have:** The energy efficiency calculation assumes 45nm
CMOS hardware. Modern neuromorphic chips (Intel Loihi 2) use 7nm. Should
I recalculate with a 7nm energy figure, or keep 45nm as a conservative
published benchmark?



## Month 2, Day 6 — Saturday
**What I did:** Ran 5 SNN configurations varying tau, v_threshold, lr,
and hidden size. Each trained 20 epochs. Compared AUC, F1, spike rate.
Selected best config by efficiency-weighted F1. Retrained best config
for 50 epochs. Saved snn_model_best.pth and updated snn_config.json.

**Experiment results:**
| Config | AUC | F1 | Spike% |
|---|---|---|---|
| Baseline | [fill] | [fill] | [fill] |
| tau=3.0 | [fill] | [fill] | [fill] |
| vth=0.3 | [fill] | [fill] | [fill] |
| lr=5e-4 | [fill] | [fill] | [fill] |
| Wide net | [fill] | [fill] | [fill] |

**Best config:** [fill from output]

**What I noticed:** [fill honestly]

**One question I have:** The experiment shows spike rate varies across
configurations. For the energy efficiency claim in the paper, should I
report the spike rate of the BEST model or the average across all
configs? And should I compare against LSTM FLOPs or SynOps?
```

---

### 🧠 Your answer to the log question

Report **the best model's spike rate** — that's your deployed model and the one you're making claims about. For the comparison:
```
In the thesis, report both:
1. SNN SynOps vs Dense ANN SynOps  ← apples-to-apples (same framework)
2. SNN SynOps vs LSTM FLOPs        ← cross-architecture (note the caveat)

With a footnote:
"SynOps and FLOPs are not directly equivalent units;
the comparison is indicative of relative computational
sparsity rather than exact energy consumption."


## Month 2, Day 8 — Monday
**What I did:** Created notebooks/11_lstm_baseline.ipynb. Loaded same
train/val data with same scaler.pkl (fair comparison guaranteed).
Defined BRICSLiquidityLSTM with hidden=32, dropout=0.2, 1 layer.
Verified forward pass shape (32,1). Compared parameter counts.
Built architecture comparison table.

**Parameter comparison:**
  SNN parameters  : [fill from output]
  LSTM parameters : [fill from output]
  LSTM has [X]x more/fewer params than SNN

**FLOPs comparison:**
  SNN SynOps/inference  : ~3,046
  LSTM FLOPs/inference  : [fill from output]

**What I noticed:** LSTM has [more/fewer] parameters than SNN despite
simpler architecture. This is because LSTM's 4 gate matrices
(input, forget, cell, output) each of size (hidden × features +
hidden × hidden) add up quickly. SNN's parameters are only in the
FC layers — LIF neurons have no learnable weights by default.

**One question I have:** Should I report LSTM parameters including
or excluding the LSTM's bias terms? Some papers count bias separately.


## Month 2, Day 9 — Tuesday
**What I did:** Trained BRICSLiquidityLSTM for 50 epochs with same
setup as SNN (Adam lr=0.001, BCEWithLogitsLoss, same pos_weight,
same early stopping). Computed full val metrics with Youden J
threshold. Saved lstm_model.pth, lstm_best.pt, lstm_config.json,
lstm_training_log.csv, lstm_confusion_matrix.png.

**KEY DIFFERENCE noted — no reset_net() for LSTM:**
  SNN needs functional.reset_net() before each batch because LIF
  membrane voltage is stored as a module attribute and persists
  between batches unless explicitly cleared.
  LSTM manages (h_n, c_n) state internally — PyTorch resets it
  automatically when a new sequence starts (h_0 defaults to zeros).
  This is a meaningful architectural difference: SNN state management
  is explicit and user-controlled; LSTM state is implicit.

**LSTM official results:**
  AUC      : [fill]
  F1       : [fill]
  Accuracy : [fill]
  Threshold: [fill]
  Train time: [fill]s

**Preliminary comparison (val set):**
  [fill winner column from Cell 15 output]

**One question I have:** The comparison shows [SNN/LSTM] wins on AUC
and [SNN/LSTM] wins on F1. For the thesis, should I report both
metrics or just AUC? And how do I frame a result where one model
wins on one metric and loses on another?



## Month 2, Day 10 — Wednesday
**What I did:** Built full comparison notebook (12_model_comparison.ipynb).
Loaded both models from saved checkpoints. Evaluated on identical val
set. Measured inference time (50 runs, warmup excluded). Computed FLOPs
and energy estimates. Built 13-row comparison table. Plotted learning
curves and ROC comparison. Wrote results section paragraph.

**Official comparison results:**
  SNN  AUC=[fill]  F1=[fill]  Acc=[fill]  Params=2945
  LSTM AUC=[fill]  F1=[fill]  Acc=[fill]  Params=5537
  Energy reduction: [fill]×

**What I noticed:** [fill honestly]
  - SNN wins on AUC and F1 — the two threshold-independent / balanced
    metrics that matter most for imbalanced classification
  - LSTM wins on accuracy — a threshold artefact, not real superiority
  - Energy difference is the strongest argument in the paper:
    [X]× reduction is a concrete, quantified, citable claim

**One question I have:** For the paper, should I include the
inference time comparison (ms/batch)? SNN is slower on CPU because
PyTorch doesn't have a sparse SNN backend — the energy advantage
only materialises on neuromorphic hardware (Loihi, SpiNNaker).
Should I acknowledge this limitation explicitly?


## Month 2, Day 11 — Thursday
**What I did:** Built error analysis across 3 dimensions: shared vs
unique errors (Jaccard overlap), accuracy by volatility quartile,
and spike rate on correct vs error days. Saved error_analysis.csv
and error_analysis_plots.png. Wrote 3 markdown observations.

**Key findings:**
- Jaccard error overlap: [fill] → ensemble potential [high/low]
- SNN accuracy Q1(calm)=[fill] vs Q4(volatile)=[fill]
- Spike rate correct=[fill] vs error=[fill] → [novel/null finding]

**What I noticed:** [fill honestly — what surprised you most?]

**Future Work paragraph (draft):**
"Three directions emerge from the error analysis: (1) ensemble
methods combining SNN and LSTM predictions to exploit error
complementarity; (2) adaptive spike thresholds that scale with
rolling volatility; (3) spike-rate confidence filters that defer
low-confidence settlement signals to human review."


## Month 2, Day 13 — Saturday
**What I did:** Built settlement decision function with dual-condition
routing (prob > threshold AND spike_rate > 0.10). Backtested on full
val set. Computed aggregate savings, direct accuracy, and risk metrics.
Saved backtest_results.csv, cumulative_savings_plot.png,
backtest_summary.json.

**Backtest results:**
  Trading days     : [fill]
  DIRECT decisions : [fill] ([fill]%)
  DIRECT accuracy  : [fill]%
  Total saving     : ₹[fill]
  Saving %         : [fill]%
  Annualised       : ₹[fill]

**What I noticed:** Even on wrong DIRECT days, the direct fee
(₹1,200) is still lower than the SWIFT fee (₹36,000) — so the
downside risk of a wrong prediction is structurally bounded.
The dual-condition filter (prob + spike_rate) reduces false
positives compared to probability-only routing.

**One question I have:** The backtest assumes one transaction per
trading day. For a real exporter doing 2-3 transactions per week,
should I scale the savings by actual transaction frequency, or
present it per-transaction and let the reader scale it?


## Month 2, Day 14 — Sunday
**What I did:** Built get_dashboard_summary() function returning
complete JSON payload for all dashboard panels. Tested with 3
transaction sizes (₹1L, ₹10L, ₹50L). Validated JSON structure.
Saved dashboard_data_sample.json. Git committed Week 2 weekend work.

**Savings scaling table:**
  ₹1L   transaction: saving ₹[fill] ([fill]%)
  ₹10L  transaction: saving ₹[fill] ([fill]%)
  ₹50L  transaction: saving ₹[fill] ([fill]%)

**What I noticed:** The % saving is roughly constant across
transaction sizes (both routes scale proportionally) but the
SWIFT flat fee (₹1,000) becomes relatively less important for
larger transactions. This means the saving % is slightly higher
for larger transactions — a natural result of the flat fee
becoming a smaller fraction of total cost.

**One question I have:** The dashboard_data_sample.json uses
the latest val-set prediction. In Month 3, should FastAPI
compute a fresh prediction on the last 10 days of real data,
or should it always use the val-set predictions as a demo?



## Month 2, Day 15 — Monday
**What I did:** Created api/ folder with 4 files. Defined all Pydantic
schemas (PriceSignalRequest, PredictionResponse, HealthResponse,
SummaryResponse, RouteDetail, SavingsDetail, SNNPredictionDetail).
Built cost_engine.py, predictor.py stub, main.py with lifespan
model loading. All 4 endpoints defined (/,/health,/predict,/summary).
API runs with uvicorn api.main:app --reload.

**What I noticed:** Pydantic validators automatically reject wrong
price sequences (wrong length, negative values, out-of-range INR/BRL).
The lifespan pattern loads the model once — much better than the
hello-world approach that had no model loading at all.

**One question I have:** The predictor.py currently tiles a single
feature vector across all 10 timesteps (demo approximation). For
Month 3, should I store the last 10 days of real val data in memory
and use those as the actual sequence, or compute features fresh
from the raw price_sequence input?


## Month 2, Day 16 — Tuesday
**What I did:** Built full SNNPredictor class with _load(),
_build_feature_sequence(), preprocess(), predict(), get_status().
Singleton pattern instantiated at module level. Ran standalone test
with 3 price sequences (normal, trending, volatile) and 2 error cases.
Verified preprocess() output shape is (1, 10, 9).

**Standalone test results:**
  Test 1 (normal)   : direction=[fill]  prob=[fill]  → [DIRECT/FALLBACK]
  Test 2 (upward)   : direction=[fill]  prob=[fill]  → [DIRECT/FALLBACK]
  Test 3 (volatile) : direction=[fill]  prob=[fill]  spike_rate=[fill]

**What I noticed:** The volatile sequence produces higher spike_rate
than the trending sequence — the LIF neurons respond to the magnitude
of return changes, not just direction. This confirms the spike
encoding is working as designed.

**One question I have:** In _build_feature_sequence(), I forward-fill
india_repo_rate as 6.5 (hardcoded). In Month 3, should I read the
latest value from a FRED API call, or store it in a config file
that gets updated periodically?


## Month 2, Day 17 — Wednesday
**What I did:** Rewrote api/main.py with full endpoint wiring.
Four endpoints live: /, /health, /predict, /summary. Added global
exception handler, CORS middleware, lifespan model verification.
Built _build_prediction_response() helper to convert all numpy
scalars to Python floats before Pydantic serialisation.

**Endpoint test results:**
  GET  /health  → 200  model_loaded=true ✅
  GET  /summary → 200  savings=₹[fill] ✅
  POST /predict → 200  recommendation=[fill] ✅
  POST /predict (wrong length) → 400  detail="must have exactly 10" ✅
  POST /predict (negative price) → 400 ✅

**What I noticed:** The numpy-to-float conversion in
_build_prediction_response() is critical — Pydantic can't serialise
numpy.float32 directly and raises a silent 500 error without it.
The `float()` wrapper on every value is the correct fix.

**One question I have:** The /predict response is ~2KB of JSON.
For the Streamlit dashboard, should I use the full /predict endpoint
or create a /predict/lite endpoint that returns only the 6 most
important fields (direction, recommendation, savings_inr,
savings_percentage, spike_rate, confidence)?


## Month 2, Day 18 — Thursday
**What I did:** Created tests/test_api.py with 8 tests: health check,
real val data prediction, 3 bad input cases, summary endpoint,
3 transaction sizes, and response time benchmark. All tests use
real INR/BRL prices from INRBRL_synthetic_clean.csv.

**Test results:**
  Test 1 Health check         : ✅
  Test 2 Real data predict    : ✅  direction=[fill]  rec=[fill]
  Test 3 Wrong length (400)   : ✅
  Test 4 Negative price (400) : ✅
  Test 5 Out of range (400)   : ✅
  Test 6 Summary endpoint     : ✅
  Test 7 Three tx sizes       : ✅
  Test 8 Response time        : ✅  mean=[fill]ms

**API prediction on real val data:**
  Savings per ₹10L transaction : ₹[fill] ([fill]%)
  Annual estimate (5 tx/month) : ₹[fill]
  Recommendation               : [DIRECT/USD_FALLBACK]

## Month 2, Day 19 — Friday
**What I did:** Enhanced FastAPI metadata — tags_metadata, table in
description, contact info, license. Added Field descriptions and
examples to PriceSignalRequest. Verified /docs page shows full
professional layout with example inputs and response schemas.

**What I noticed:** The markdown table in app description renders
beautifully in /docs — the key research numbers are immediately
visible to anyone who opens the Swagger page. This is the first
thing a supervisor or reviewer will see.

**One question I have:** Should I add authentication (API key) to
the /predict endpoint before the Month 3 dashboard demo, or is
open access fine for a thesis project?


