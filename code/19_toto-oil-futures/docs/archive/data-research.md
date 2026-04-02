# Data Research — What Can We Get?

Date: 2026-04-02

## yfinance Test Results (actual data pulled today)

**40/40 tickers returned data.** Tested futures, ETFs, and sector stocks.

### Futures (16 tickers)
- Most go back to ~2000, giving **~6,400+ rows each**
- Clean (no gaps): CL=F, NG=F, HO=F, RB=F, GC=F, SI=F, HG=F, ZS=F, KC=F, CT=F, SB=F
- Messy (gaps): PL=F (22 gaps), PA=F (21 gaps) — skip these for MVP
- Minor gaps: ZC=F (2), ZW=F (1), BZ=F (1) — usable

### ETFs (21 tickers)
- **Zero gaps in all ETFs** — very clean
- GLD longest: 2004-11-18 (5,375 rows)
- 5 are delisted (JO, BAL, SGG, COW, NIB) — data stops 2023-07
- PDBC shortest active: from 2014

### Intraday
- 1h bars: ~2 years back (April 2024)
- 5m bars: ~2 months back (Feb 2026)
- Futures have more bars (24h trading) vs ETFs (market hours only)

### Totals from yfinance alone
- **198,376 total data rows**
- **7,165 unique trading days** (1997–2026)

## Expanded Ticker Universe (24 clean futures)

```
CL=F (Crude Oil WTI), BZ=F (Brent), NG=F (Natural Gas), HO=F (Heating Oil),
RB=F (Gasoline), GC=F (Gold), SI=F (Silver), HG=F (Copper),
ZC=F (Corn), ZW=F (Wheat), ZS=F (Soybeans), ZM=F (Soybean Meal),
ZL=F (Soybean Oil), KC=F (Coffee), SB=F (Sugar), CC=F (Cocoa),
CT=F (Cotton), OJ=F (Orange Juice), LBS=F (Lumber),
LE=F (Live Cattle), HE=F (Lean Hogs), ZR=F (Rice)
```

Plus ETFs for cleaner data: USO, UNG, GLD, SLV, DBA, DBC, CPER, WEAT, CORN, SOYB, UGA, BNO, PALL, PPLT

## Other Free Sources

| Source | # Commodities | Frequency | History | Notes |
|--------|--------------|-----------|---------|-------|
| yfinance | 24+ futures, 16+ ETFs | Daily | 10-25yr | Free, no key, our primary |
| Kaggle (mattiuzc) | ~30 | Daily | 30-50yr | Clean CSVs, longest history |
| FRED | 6-10 | Daily | 30-50yr | Energy + metals only at daily |
| EIA | 10-15 energy | Daily | 30yr+ | Best quality for energy |
| World Bank | 70+ | Monthly only | 60yr+ | Too low freq for us |

## Do We Have Enough to Fine-Tune?

### The math
- 24 futures × ~6,000 rows each = **~144,000 daily data points**
- Add 16 ETFs × ~4,000 rows each = **+64,000 more**
- Total: **~200,000+ data points across ~40 series**

### What Toto needs
- Toto's fine-tuning config samples 100 windows per series per epoch
- With 40 series × 100 windows = 4,000 training samples per epoch
- Default fine-tuning is 1,400 steps at batch_size=16 = ~22,400 samples = ~5-6 epochs
- **This is feasible.** Not massive, but the Toto authors designed fine-tuning to work with modest data.

### Risk: domain gap
- Toto was pre-trained on IT/observability metrics (CPU, latency, throughput)
- Commodities are structurally different (non-stationary, fat tails, regime changes)
- Fine-tuning has high potential upside precisely BECAUSE of this gap
- But 40 series is thin — overfitting is a real risk

### Mitigation strategies
1. **Use ALL tickers, not just oil** — more series = more robust fine-tuning
2. **Include ETFs alongside futures** — doubles the series count, cleaner data
3. **Use returns not prices** — removes non-stationarity, reduces domain gap
4. **Short fine-tuning** — fewer steps to avoid overfitting (Toto defaults are conservative)
5. **Always compare to zero-shot** — if fine-tuning doesn't beat zero-shot, we stop

## Decision: Futures vs ETFs vs Both

| | Futures | ETFs | Both |
|--|---------|------|------|
| Pros | Real commodity exposure, 24h data | No roll gaps, clean, standardized | Most data, most robust |
| Cons | Roll artifacts, gaps in some | Shorter history, management fees affect price | Mixing apples and oranges? |
| Verdict | Primary target | Supplement | **Use both for fine-tuning, evaluate on futures** |

## Recommendation

**Use yfinance as primary source.** Pull everything — 24 futures + 16 ETFs = 40 series.
Fine-tune on all 40 series. Evaluate/forecast on the futures (the real signal).

**Don't need Kaggle/FRED for MVP.** yfinance gives us enough. Kaggle is a good backup for longer history if we need it later.

**Returns, not prices.** This is the key decision — train on daily returns:
- Removes rollover artifacts
- Makes all series comparable in scale  
- Aligns with directional accuracy as primary metric
- Toto's instance normalization helps too, but returns are cleaner

## Fine-Tuning Time Estimate
- T4 GPU (Colab free tier)
- 1,400 steps, batch_size=16
- Toto is 151M params — small by LLM standards
- **Estimate: 15-30 minutes per fine-tune run**
- Autoresearch overnight (Phase 2): could run 30-50 experiments in 8 hours
