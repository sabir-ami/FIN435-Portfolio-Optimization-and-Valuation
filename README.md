# Multi-Sector Stock Price Prediction & Portfolio Optimization (NLP Sentiment + Denoising + ML)

An end-to-end machine learning pipeline that combines *news sentiment (FinBERT)*, *signal denoising*, and *nine forecasting models* to predict short-horizon intraday prices for four stocks across four sectors, then turns those forecasts into a *buy/sell portfolio allocation*.

| Ticker | Sector |
|--------|--------|
| AAPL | Technology |
| JPM | Financials |
| XOM | Energy |
| JNJ | Healthcare |

**Important:** This is a methods and engineering project, not a trading strategy. See [Limitations](#limitations-read-this-first) before interpreting any number below.


---

## Pipeline Overview

1. Price download      -> 1-minute OHLCV data per stock (yfinance)
2. Prediction          -> sentiment + denoising + 9 models -> 150-minute forecast
3. Portfolio           -> combine forecasts, classify buy/sell, allocate capital

Each stock has its own download and prediction notebook (same code, different TARGET_COLUMN), plus one shared portfolio notebook.

### 1. Price download
- 1-minute bars from yfinance (pinned to 0.2.66) for Sept 14-18, 2026, about 1,950 rows per stock.
- Requests are made one day at a time, with weekend days skipped, to avoid empty-data retries.
- The MultiIndex columns that yfinance returns are flattened before saving.

### 2. Prediction (per stock, T4 GPU on Colab)

| Step | What it does |
|------|--------------|
| News scraping | 50 recent headlines per stock from FinViz, with date carry-forward because FinViz prints the date only on the first headline of each day |
| Sentiment | FinBERT scores each headline (positive / negative / neutral, plus derived polarity, strength, impact) |
| Merge | pd.merge_asof(direction='backward') attaches each price row to the most recent headline at or before it, so there is no look-ahead |
| Denoising | Optuna (10 trials) searches wavelet, moving-average, and Kalman filters, maximizing SNR subject to a variance-retention constraint |
| Sequences | 30-step windows over 7 features (denoised price + 6 sentiment features) |
| Models | CNN-LSTM, Conv1D-LSTM, BiLSTM, Stacked LSTM, GRU, LightGBM, XGBoost, Transformer, TCN |
| Tuning | Optuna, 3 trials per model, optimizing adjusted R² |
| Validation | Walk-forward (step 300, up to 20 steps) |
| Overfit check | Train vs. validation adjusted R²; models with gap > 0.2 are excluded |
| Forecast | Best surviving model produces a 150-step autoregressive forecast seeded from the most recent 30 rows |

### 3. Portfolio optimization
- The four forecasts are merged on timestamp into a single table.
- Each asset is classified by timing: if its forecast minimum comes before its maximum it is a *buy*, otherwise a *sell*.
- Capital is split between the buy and sell groups, and within each group by *inverse standard deviation* (lower forecast volatility gets more weight).
- Whole-share trades are computed from a $10,000 budget.

---

## Results

### Denoising (SNR and variance retained)

| Stock | SNR (dB) | Variance retained |
|-------|---------:|------------------:|
| AAPL | 4.20 | 99.8% |
| JPM | 35.23 | 99.8% |
| XOM | 39.04 | 100.0% |
| JNJ | 41.17 | 99.9% |

### Model comparison (validation adjusted R²)

| Stock | LightGBM | XGBoost | GRU | Selected |
|-------|---------:|--------:|----:|----------|
| AAPL | 0.984 | 0.983 | 0.968 | LightGBM |
| JPM | 0.981 | 0.974 | 0.947 | XGBoost |
| XOM | 0.985 | 0.982 | 0.922 | XGBoost |
| JNJ | 0.910 | 0.921 | 0.878 | XGBoost |

Tree-based models (LightGBM, XGBoost) consistently came out on top and passed the overfitting check. Several deep models failed it, most notably on XOM, where some produced extremely negative validation R². The final selection for JPM, XOM, and JNJ was made after the train/validation gap filter, which is why XGBoost is chosen over LightGBM for JPM and XOM despite a slightly lower raw score.

### Portfolio (150-minute window, not annualized)

| Metric | Value |
|--------|------:|
| Buy: JNJ (13 shares), JPM (1 share) | return 0.230%, Sharpe 3.41 |
| Sell: AAPL (11 shares), XOM (10 shares) | return 0.215%, Sharpe 3.97 |
| Overall return | 0.221% |
| Overall volatility | 0.000843 |
| Overall Sharpe | 2.56 |
| Unallocated cash | $817.51 |

These are *period-level* figures over a 150-minute forecast window. They are deliberately not annualized (see bug 4 below).

---

## Key Design Decisions

- *Variance floor on denoising.* Maximizing SNR alone rewards over-smoothing and can flatten the price signal almost completely. Any denoising candidate that keeps less than 50% of the original standard deviation is disqualified before SNR is compared.
- *Time-based sentiment alignment.* Headlines are matched to price rows by timestamp (merge_asof, backward), never by row position, so each price bar only sees news published at or before it.
- *All features reach every model.* Sequences are flattened with reshape(n, -1) for the tree models so the sentiment features are used alongside price.
- *Walk-forward validation.* Models are evaluated on successive forward windows rather than a single random split, and the session is cleared between models (not between steps) to keep GPU memory under control.
- *Forecast seeded from the latest data.* The autoregressive forecast starts from the most recent 30 rows.
- *No annualization.* The forecast horizon is 150 minutes, so annualizing would compound a sub-day return as if it were a full trading day and produce meaningless figures. Portfolio metrics are reported per window.

---

## Limitations (read this first)

I want to be explicit about what these results do and do not show.

- *Scaler leakage is not fixed.* MinMaxScaler is fit on the full price series before the train/validation split, so the scaler has seen validation-period values. This likely inflates validation R² somewhat.
- *No ground-truth backtest.* The 150-minute forecast extends beyond the end of the data, so it cannot be checked against real prices. All R², return, and Sharpe figures describe fit to historical patterns, not demonstrated predictive skill.
- *Single split, single seed, single window.* Nothing was cross-validated across multiple time periods or random seeds, so the model rankings may not be stable.
- *Thin sentiment signal.* Only 50 headlines per stock, unevenly distributed in time. For AAPL only about half of the price rows have a real sentiment match. The contribution of sentiment was never isolated with an ablation test.
- *Heuristic thresholds.* The 0.2 overfitting-gap cutoff and the 0.5 variance-retention floor were chosen by judgment, not tuned.
- *Very high R² on prices.* Adjusted R² around 0.98 on a smooth, autocorrelated price series is easy to achieve and says little about forecasting returns. Evaluating on returns or directional accuracy would be a more meaningful test.
- *Portfolio metrics are illustrative.* A Sharpe ratio computed over one 150-minute forecast window is not a performance claim.

### Possible next steps
- Fit the scaler on the training split only
- Rolling multi-window validation and multiple seeds
- Evaluate on returns and directional accuracy against a naive baseline
- Sentiment ablation study
- More headlines, with a longer price history

---

## How to Run

1. Open a download notebook in Google Colab and run it to produce the price CSV for that ticker.
2. Open the matching prediction notebook, select a *T4 GPU* runtime, set TARGET_COLUMN, upload the CSV when prompted, and run all cells. This produces <TICKER>_predictions.csv.
3. Repeat for all four tickers.
4. Open Portfolio.ipynb, upload the four prediction CSVs, and run all cells.

## Tech Stack

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers (FinBERT), LightGBM, XGBoost, Optuna, PyWavelets, yfinance, Google Colab (T4 GPU)

## Disclaimer

This project is for educational and portfolio purposes only. It is not financial advice.




# Robi Axiata PLC: Financial Analysis & Valuation

An Excel model that analyses Robi Axiata PLC's 2020-2025 financial statements, estimates its cost of capital, and values the company using four methods. Each implied share price is compared with the current market price of BDT 29.80.

## How It Was Done

1. *Historical data.* Income statement, balance sheet and cash flow statement for 2020-2025 (sheets: Income Statement-Robi, Balance Sheet-Robi, Cash Flow-Robi).
2. *Cost of capital.*
   - Effective tax rate = income tax ÷ profit before tax (TAX). It fell from 71.8% in 2020 to 17.1% in 2025.
   - Cost of debt is the average implied rate, about 0.72% (Cost of Debt).
   - Cost of equity uses the Gordon growth formula: D1 ÷ P0 + g = *49.54%*, with g = 49.08% (average net-profit growth) (Cost of Equity).
   - WACC blends the two using market cap and net debt = *26.98%*.
3. *Free cash flow.* Built as FCF, CFFA, FCFF and FCFE (FCF, CFFA, FCFF & FCFE).
4. *Forecast and discount.* Each model projects cash flows or dividends over 2025-2030, adds a terminal value (perpetual growth 0.5%), discounts to present value, and divides by 5,237,932,895 shares.

| Model | Sheet | Discount rate | Growth assumed |
|---|---|---|---|
| FCF-based DCF | CFFA | WACC | 30% |
| FCFF | FCFF Valuation | WACC | 30% |
| FCFE | FCFE Valuation | Cost of equity | 49.08% |
| Dividend Discount Model | DDM | WACC | Dividends 17.5% (net profit 49.08%) |

## Results

| Model | Implied price (BDT) | vs. market price (29.80) |
|---|---|---|
| FCF-based DCF | 24.63 | about 17% below |
| FCFE | 71.69 | about 2.4x above |
| FCFF | 124.44 | about 4.2x above |
| DDM | 1.26 | far below |

## What the Results Mean

- *Market price is in the middle of the range.* The FCF-based DCF (24.63) is close to the market price, suggesting the stock is roughly fairly valued under conservative cash flow assumptions.
- *FCFF and FCFE point to undervaluation*, but they rest on very high growth (30% and 49% a year) and a large terminal value, so they likely overstate value.
- *The DDM is low* because Robi pays small dividends (DPS 0.175 in 2025) and they are discounted at a 27% rate. It reflects payout, not earning power.
- *The wide spread (1.26 to 124.44)* shows that the valuation is highly sensitive to the cash flow definition, growth rates and discount rate. No single figure should be treated as the "true" value.

## Limitations

- FCFF and FCFE appear not to deduct capital expenditure, which inflates them.
- FCFF and FCFE equity bridges use cash, debt and securities figures that differ from the balance sheet.
- Growth rates (30% and 49%) are far above any sustainable long-run level, and the 49% also drives the cost of equity.
- The cost of debt (0.72%) is low relative to finance expense ÷ total debt (about 5%).
- 2025 gross profit does not equal revenue minus cost of sales (36.31 bn reported vs 38.65 bn calculated).
- The Cost of Debt sheet calls Robi an "FMCG business", which conflicts with the rest of the workbook.

## Tools

Microsoft Excel; discounted cash flow, WACC and dividend discount modeling.

## Disclaimer

For academic and educational purposes only. Results depend on the model's assumptions and should not be treated as investment advice.



# Quantitative Finance Projects: Valuation, Portfolio Optimization and ML Price Prediction

Three finance projects in one repository:

1. *Robi Axiata PLC: Financial Analysis & Valuation* (Excel). Estimates the company's cost of capital and intrinsic share value using four valuation methods.
2. *Portfolio Optimization of Four DSE Stocks* (Excel). Finds the best-Sharpe-ratio portfolio of ACI, APEXFOOT, BXPHARMA and BATBC from real Dhaka Stock Exchange (DSE) prices.
3. *Multi-Sector Stock Price Prediction & Portfolio Optimization* (Python). Combines news sentiment, signal denoising and nine forecasting models to predict intraday prices for AAPL, JPM, XOM and JNJ, then builds a buy/sell portfolio.

---

# Project 1: Robi Axiata PLC: Financial Analysis & Valuation

## What This Project Is About

An Excel model that analyses Robi Axiata PLC's 2020-2025 financial statements, estimates its cost of capital, and values the company with four methods. Each implied share price is compared with the current market price of BDT 29.80. The workbook (Robi.xlsx) has sheets for the three financial statements, cost of debt, cost of equity, tax rate, free cash flow (FCF, CFFA, FCFF & FCFE) and four valuation sheets (CFFA, FCFF Valuation, FCFE Valuation, DDM).

## How It Was Done

1. *Historical data.* Income statement, balance sheet and cash flow statement for 2020-2025.
2. *Cost of capital.*
   - Effective tax rate = income tax ÷ profit before tax. It fell from 71.8% in 2020 to 17.1% in 2025.
   - Cost of debt is the average implied rate, about 0.72%.
   - Cost of equity uses the Gordon growth formula: D1 ÷ P0 + g = *49.54%*, with g = 49.08% (average net-profit growth).
   - WACC blends the two using market cap and net debt = *26.98%*.
3. *Free cash flow.* Built as FCF, CFFA, FCFF and FCFE.
4. *Forecast and discount.* Each model projects cash flows or dividends over 2025-2030, adds a terminal value (perpetual growth 0.5%), discounts to present value, and divides by 5,237,932,895 shares.

| Model | Sheet | Discount rate | Growth assumed |
|---|---|---|---|
| FCF-based DCF | CFFA | WACC | 30% |
| FCFF | FCFF Valuation | WACC | 30% |
| FCFE | FCFE Valuation | Cost of equity | 49.08% |
| Dividend Discount Model | DDM | WACC | Dividends 17.5% (net profit 49.08%) |

## Results

| Model | Implied price (BDT) | vs. market price (29.80) |
|---|---|---|
| FCF-based DCF | 24.63 | about 17% below |
| FCFE | 71.69 | about 2.4x above |
| FCFF | 124.44 | about 4.2x above |
| DDM | 1.26 | far below |

## What the Results Mean

- *The FCF-based DCF (24.63) is close to the market price.* Under conservative cash flow assumptions, the stock looks roughly fairly valued.
- *FCFF and FCFE point to undervaluation*, but they rest on very high growth (30% and 49% a year) and a large terminal value, so they likely overstate value.
- *The DDM is low* because Robi pays small dividends (DPS 0.175 in 2025) and they are discounted at a 27% rate. It reflects payout, not earning power.
- *The wide spread (1.26 to 124.44)* shows that the valuation is highly sensitive to the cash flow definition, growth rates and discount rate. No single figure should be treated as the "true" value.

## Limitations

- FCFF and FCFE appear not to deduct capital expenditure, which inflates them.
- FCFF and FCFE equity bridges use cash, debt and securities figures that differ from the balance sheet.
- Growth rates (30% and 49%) are far above any sustainable long-run level, and the 49% also drives the cost of equity.
- The cost of debt (0.72%) is low relative to finance expense ÷ total debt (about 5%).
- 2025 gross profit does not equal revenue minus cost of sales (36.31 bn reported vs 38.65 bn calculated).
- The Cost of Debt sheet calls Robi an "FMCG business", which conflicts with the rest of the workbook.

---

# Project 2: Portfolio Optimization of Four DSE Stocks

## What This Project Is About

An Excel model that builds a four-stock portfolio from real DSE price data. It asks how money should be split across ACI, APEXFOOT, BXPHARMA and BATBC to get the best return for the risk taken. The best split is the portfolio with the highest Sharpe ratio, found by simulating 234 random portfolios.

*Data:* Kaggle, Dhaka Stock Exchange Historical Data (1999-2025), covering 235 trading days from 2024-04-08 to 2025-04-08. The workbook has one sheet per stock (aci, apexfoot, bxpharma, batbc) and an analysis sheet (Sheet1). A few halted-trading days were patched by copying that day's close into open/high/low.

## How It Was Done

1. Closing prices are pulled from the four data sheets.
2. Daily log returns are calculated: LN(price yesterday ÷ price day before). This gives 234 returns per stock.
3. The risk-free rate (9.75% ÷ 236, about 0.0413% per day) is subtracted from each return.
4. A 4×4 covariance matrix is built using VAR.S and COVARIANCE.S.
5. 234 random sets of weights are generated and normalised to sum to 100%. They are frozen as fixed values, so results are repeatable.
6. For each portfolio:
   - Return = SUMPRODUCT(weights, average returns)
   - Variance = MMULT(MMULT(w, Σ), TRANSPOSE(w))
   - Sharpe ratio = (return − risk-free rate) ÷ √variance
7. MAX finds the highest Sharpe ratio, and VLOOKUP returns that portfolio's return, variance and weights.

## Results

*Individual stocks (2024-04-08 to 2025-04-08)*

| Stock | Price change | Avg daily log return |
|---|---|---|
| ACI | 155.2 to 200.7 (+29.3%) | +0.110% |
| APEXFOOT | 248.8 to 214.0 (-14.0%) | -0.064% |
| BXPHARMA | 120.0 to 109.2 (-9.0%) | -0.040% |
| BATBC | 405.6 to 321.8 (-20.7%) | -0.099% |

*Maximum Sharpe portfolio*

| Metric | Result |
|---|---|
| Sharpe ratio (daily) | 0.00764 |
| Portfolio return (daily) | 0.0578% |
| Portfolio variance (daily) | 0.000467 |
| Weights | ACI 70.49%, APEXFOOT 8.20%, BXPHARMA 11.48%, BATBC 9.84% |

## What the Results Mean

- *The best portfolio is mostly ACI (about 70%).* ACI was the only stock with a positive average return, so the model favours it. The other three lost value and get small weights.
- *Few portfolios beat the risk-free rate.* Only 2 of 234 simulated portfolios had a positive Sharpe ratio, and the best is small. Over this period, these four stocks did not reliably beat the risk-free rate.
- *The best-Sharpe portfolio is not the lowest-risk one.* The lowest-variance portfolio (0.000195) had a negative return.
- *The stocks tended to move together.* All pairwise covariances are positive, so diversification helped only partly.
- *The result is a one-year snapshot.* It depends on ACI's rally and may not repeat.

## Limitations

- The best portfolio comes from random sampling, not a true optimiser such as Solver.
- It rests on a single year of data.
- The Sharpe ratio is daily, not annualised.
- The 9.75% risk-free rate is hardcoded and its source is not stated. The summary box divides it by 235, while the main calculation divides by 236.

---

# Project 3: Multi-Sector Stock Price Prediction & Portfolio Optimization

## What This Project Is About

An end-to-end machine learning pipeline that combines *news sentiment (FinBERT)*, *signal denoising* and *nine forecasting models* to predict short-horizon intraday prices for four stocks across four sectors. It then turns the forecasts into a *buy/sell portfolio allocation*.

| Ticker | Sector |
|--------|--------|
| AAPL | Technology |
| JPM | Financials |
| XOM | Energy |
| JNJ | Healthcare |

**Important:** This is a methods and engineering project, not a trading strategy. Read the [Limitations](#limitations-read-this-first) before interpreting any number below.


## How It Was Done

1. Price download  ->  1-minute OHLCV data per stock (yfinance)
2. Prediction      ->  sentiment + denoising + 9 models -> 150-minute forecast
3. Portfolio       ->  combine forecasts, classify buy/sell, allocate capital

Each stock has its own download and prediction notebook (same code, different TARGET_COLUMN), plus one shared portfolio notebook.

*1. Price download*
- 1-minute bars from yfinance (pinned to 0.2.66) for Sept 14-18, 2026, about 1,950 rows per stock.
- Requests are made one day at a time, with weekend days skipped, to avoid empty-data retries.
- The MultiIndex columns that yfinance returns are flattened before saving.

*2. Prediction (per stock, T4 GPU on Colab)*

| Step | What it does |
|------|--------------|
| News scraping | 50 recent headlines per stock from FinViz, with date carry-forward because FinViz prints the date only on the first headline of each day |
| Sentiment | FinBERT scores each headline (positive / negative / neutral, plus derived polarity, strength, impact) |
| Merge | pd.merge_asof(direction='backward') attaches each price row to the most recent headline at or before it, so there is no look-ahead |
| Denoising | Optuna (10 trials) searches wavelet, moving-average and Kalman filters, maximizing SNR subject to a variance-retention constraint |
| Sequences | 30-step windows over 7 features (denoised price + 6 sentiment features) |
| Models | CNN-LSTM, Conv1D-LSTM, BiLSTM, Stacked LSTM, GRU, LightGBM, XGBoost, Transformer, TCN |
| Tuning | Optuna, 3 trials per model, optimizing adjusted R² |
| Validation | Walk-forward (step 300, up to 20 steps) |
| Overfit check | Train vs. validation adjusted R²; models with a gap above 0.2 are excluded |
| Forecast | The best surviving model produces a 150-step autoregressive forecast seeded from the most recent 30 rows |

*3. Portfolio optimization*
- The four forecasts are merged on timestamp into a single table.
- Each asset is classified by timing: if its forecast minimum comes before its maximum it is a *buy*, otherwise a *sell*.
- Capital is split between the buy and sell groups, and within each group by *inverse standard deviation* (lower forecast volatility gets more weight).
- Whole-share trades are computed from a $10,000 budget.

*Key design decisions*
- *Variance floor on denoising.* Maximizing SNR alone rewards over-smoothing. Any candidate that keeps less than 50% of the original standard deviation is disqualified.
- *Time-based sentiment alignment.* Headlines are matched to price rows by timestamp, never by row position, so each price bar only sees news published at or before it.
- *All features reach every model.* Sequences are flattened with reshape(n, -1) for the tree models so the sentiment features are used alongside price.
- *Walk-forward validation.* Models are evaluated on successive forward windows rather than a single random split.
- *No annualization.* The horizon is 150 minutes, so annualizing would compound a sub-day return as if it were a full trading day. Portfolio metrics are reported per window.

## Results

*Denoising*

| Stock | SNR (dB) | Variance retained |
|-------|---------:|------------------:|
| AAPL | 4.20 | 99.8% |
| JPM | 35.23 | 99.8% |
| XOM | 39.04 | 100.0% |
| JNJ | 41.17 | 99.9% |

*Model comparison (validation adjusted R²)*

| Stock | LightGBM | XGBoost | GRU | Selected |
|-------|---------:|--------:|----:|----------|
| AAPL | 0.984 | 0.983 | 0.968 | LightGBM |
| JPM | 0.981 | 0.974 | 0.947 | XGBoost |
| XOM | 0.985 | 0.982 | 0.922 | XGBoost |
| JNJ | 0.910 | 0.921 | 0.878 | XGBoost |

*Portfolio (150-minute window, not annualized)*

| Metric | Value |
|--------|------:|
| Buy: JNJ (13 shares), JPM (1 share) | return 0.230%, Sharpe 3.41 |
| Sell: AAPL (11 shares), XOM (10 shares) | return 0.215%, Sharpe 3.97 |
| Overall return | 0.221% |
| Overall volatility | 0.000843 |
| Overall Sharpe | 2.56 |
| Unallocated cash | $817.51 |

## What the Results Mean

- *Tree-based models won.* LightGBM and XGBoost consistently ranked highest and passed the overfitting check. Several deep models failed it, most notably on XOM, where some produced extremely negative validation R².
- *XGBoost is chosen over LightGBM for JPM and XOM* despite a slightly lower raw score, because the selection was made after the train/validation gap filter.
- *The portfolio figures are illustrative.* A Sharpe ratio over one 150-minute forecast window describes the forecast, not demonstrated performance.

## Limitations (read this first)

- *Scaler leakage is not fixed.* MinMaxScaler is fit on the full price series before the train/validation split, which likely inflates validation R².
- *No ground-truth backtest.* The 150-minute forecast extends beyond the end of the data, so it cannot be checked against real prices. All R², return and Sharpe figures describe fit to historical patterns, not predictive skill.
- *Single split, single seed, single window.* Model rankings may not be stable.
- *Thin sentiment signal.* Only 50 headlines per stock, unevenly distributed in time. For AAPL only about half of the price rows have a real sentiment match. Sentiment's contribution was never isolated with an ablation test.
- *Heuristic thresholds.* The 0.2 overfitting-gap cutoff and the 0.5 variance-retention floor were chosen by judgment.
- *Very high R² on prices.* An adjusted R² around 0.98 on a smooth, autocorrelated price series is easy to achieve and says little about forecasting returns.

*Possible next steps:* fit the scaler on the training split only; rolling multi-window validation and multiple seeds; evaluate on returns and directional accuracy against a naive baseline; a sentiment ablation study; more headlines and a longer price history.

## How to Run

1. Open a download notebook in Google Colab and run it to produce the price CSV for that ticker.
2. Open the matching prediction notebook, select a *T4 GPU* runtime, set TARGET_COLUMN, upload the CSV when prompted, and run all cells. This produces <TICKER>_predictions.csv.
3. Repeat for all four tickers.
4. Open Portfolio.ipynb, upload the four prediction CSVs, and run all cells.

---

## Tools

- *Projects 1 and 2:* Microsoft Excel; discounted cash flow, WACC and dividend discount modeling; Modern Portfolio Theory (covariance matrix, Sharpe ratio).
- *Project 3:* Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers (FinBERT), LightGBM, XGBoost, Optuna, PyWavelets, yfinance, Google Colab (T4 GPU).

## Disclaimer

All three projects are for academic, educational and portfolio purposes only. Results depend on the assumptions and data used and are not financial advice.


