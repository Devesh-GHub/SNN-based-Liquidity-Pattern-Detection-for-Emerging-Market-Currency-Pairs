# Deployment Notes
**Date:** Month 2, Day 20
**Platform:** Render.com (free tier)
**Live URL:** https://snn-based-liquidity-pattern-detection-i1t5.onrender.com/

## Endpoints (live)
| Endpoint | URL |
|---|---|
| Health | https://snn-based-liquidity-pattern-detection-i1t5.onrender.com//health |
| Predict | https://snn-based-liquidity-pattern-detection-i1t5.onrender.com//predict |
| Summary | https://snn-based-liquidity-pattern-detection-i1t5.onrender.com//summary |
| Docs    | https://snn-based-liquidity-pattern-detection-i1t5.onrender.com//docs |

## Render free tier limitations
- Spins down after 15 minutes of inactivity
- First request after sleep takes ~30-60 seconds (cold start)
- 512MB RAM — sufficient for our 3K parameter SNN
- 0.1 CPU — inference is fast enough (~100ms)

## Cold start handling (for dashboard)
In the Streamlit dashboard (Month 3), add a loading spinner
that shows while the API wakes up:
  "Waking up the SNN model — first request may take 30 seconds..."

## What is NOT deployed
- Raw data CSVs (too large, not needed for inference)
- Training notebooks (not needed for inference)
- All outputs/ except model files

## Reproducing the deployment
1. Clone: git clone https://github.com/Devesh-GHub/SNN-based-Liquidity-Pattern-Detection-for-Emerging-Market-Currency-Pairs
2. Install: pip install -r requirements.txt
3. Run: uvicorn api.main:app --reload
4. Test: python tests/test_api.py