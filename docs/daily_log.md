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


