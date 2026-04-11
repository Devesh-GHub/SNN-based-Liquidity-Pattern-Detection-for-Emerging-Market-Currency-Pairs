"""
tests/test_api.py
=================
End-to-end API tests for the BRICS SNN Settlement API.

Run with the API server already started:
    # Terminal 1:
    uvicorn api.main:app --reload

    # Terminal 2:
    python tests/test_api.py        ← standalone
    pytest tests/test_api.py -v     ← with pytest

All tests use real data from data/processed/val_features.csv
where possible — not dummy data.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import sys
import os

BASE_URL = "http://localhost:8000"

# ── ANSI colours for terminal output ─────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg):print(f"  {RED}❌ {msg}{RESET}")
def info(msg):print(f"  {YELLOW}ℹ  {msg}{RESET}")


# ─────────────────────────────────────────────
# HELPER: get real INR/BRL prices from val set
# ─────────────────────────────────────────────

def get_real_prices(n: int = 10) -> list:
    """
    Load the last n INR/BRL closing prices from val_features.csv.

    Falls back to realistic synthetic prices if the file is not found.
    """
    val_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed",
        "INRBRL_synthetic_clean.csv"
    )

    if os.path.exists(val_path):
        # INRBRL_synthetic_clean.csv stores the date in the first (unnamed) column.
        df = pd.read_csv(val_path)
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.sort_values(first_col)

        # Prefer the actual INR/BRL synthetic cross-rate column.
        rate_col = None
        for candidate in ("inr_brl_synthetic", "inrbrl_synthetic", "inr_brl"):
            if candidate in df.columns:
                rate_col = candidate
                break

        # Fallback: any column that looks like INR/BRL.
        if rate_col is None:
            rate_col = next(
                (c for c in df.columns if "inr" in c.lower() and "brl" in c.lower()),
                None,
            )

        # Last resort: any '*close*' column.
        if rate_col is None:
            rate_col = next((c for c in df.columns if "close" in c.lower()), None)

        if rate_col and len(df) >= n:
            prices = df[rate_col].dropna().tail(n).tolist()
            if len(prices) == n and all(5 < float(p) < 100 for p in prices):
                return [round(float(p), 4) for p in prices]

    # Fallback — realistic INR/BRL rates
    info("Using fallback prices (val CSV not found or too short)")
    return [16.42, 16.38, 16.51, 16.45, 16.60,
            16.53, 16.48, 16.55, 16.61, 16.57]


# ─────────────────────────────────────────────
# TEST 1: Health check
# ─────────────────────────────────────────────

def test_health():
    """GET /health should return 200 with model_loaded=true."""
    print(f"\n{BOLD}TEST 1 — Health Check{RESET}")
    print(f"  GET {BASE_URL}/health")

    resp = requests.get(f"{BASE_URL}/health", timeout=10)

    assert resp.status_code == 200, \
        f"Expected 200, got {resp.status_code}: {resp.text}"

    data = resp.json()
    assert data["status"]       == "healthy",  f"status={data['status']}"
    assert data["model_loaded"] == True,       "model_loaded is False"
    assert "val_auc"   in data,                "missing val_auc"
    assert "uptime_seconds" in data,           "missing uptime_seconds"

    ok(f"status={data['status']}")
    ok(f"model_loaded={data['model_loaded']}")
    ok(f"model_version={data['model_version']}")
    ok(f"val_auc={data['val_auc']}")
    ok(f"uptime={data['uptime_seconds']}s")
    return data


# ─────────────────────────────────────────────
# TEST 2: Predict with real val data
# ─────────────────────────────────────────────

def test_predict_with_real_data():
    """POST /predict with real INR/BRL prices from val set."""
    print(f"\n{BOLD}TEST 2 — Predict with Real Val Data{RESET}")

    prices = get_real_prices(10)
    info(f"Prices loaded  : {prices}")
    info(f"Price range    : [{min(prices):.4f}, {max(prices):.4f}]")

    payload = {
        "price_sequence"        : prices,
        "transaction_amount_inr": 1_000_000,
        "monthly_tx_count"      : 5,
    }

    t0   = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/predict",
                         json=payload, timeout=30)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, \
        f"Expected 200, got {resp.status_code}: {resp.text}"

    result = resp.json()

    # ── Field presence checks ─────────────────────────────────────────
    required = [
        "direction", "confidence", "spike_rate", "recommendation",
        "usd_route_cost", "snn_route_cost", "savings_inr",
        "savings_percentage", "settlement_time_current",
        "settlement_time_proposed", "usd_route", "snn_route",
        "savings", "snn_prediction", "assumptions",
    ]
    for field in required:
        assert field in result, f"Missing field: {field}"

    # ── Value range checks ────────────────────────────────────────────
    assert result["direction"]   in ("UP", "DOWN"),            "Invalid direction"
    assert 0 <= result["confidence"]  <= 1,                    "confidence out of range"
    assert 0 <= result["spike_rate"]  <= 1,                    "spike_rate out of range"
    assert result["recommendation"] in ("DIRECT","USD_FALLBACK"), "Invalid recommendation"
    assert result["usd_route_cost"] > result["snn_route_cost"], "SNN should be cheaper"
    assert result["savings_inr"]    > 0,                       "savings must be positive"
    assert 0 < result["savings_percentage"] < 100,             "savings_pct out of range"

    # ── Nested structure checks ───────────────────────────────────────
    assert "total_cost_inr" in result["usd_route"],    "usd_route missing total"
    assert "total_cost_inr" in result["snn_route"],    "snn_route missing total"
    assert "amount_inr"     in result["savings"],      "savings missing amount"
    assert "probability"    in result["snn_prediction"],"snn_prediction missing prob"

    # ── Print the full result ─────────────────────────────────────────
    print(f"\n  {'='*50}")
    print(f"  {BOLD}LIVE API PREDICTION RESULT{RESET}")
    print(f"  {'='*50}")
    print(f"  Direction       : {BOLD}{result['direction']}{RESET}")
    print(f"  Confidence      : {result['confidence']:.2%}")
    print(f"  Spike Rate      : {result['spike_rate']:.2%}")
    print(f"  Recommendation  : {BOLD}{result['recommendation']}{RESET}")
    print(f"  {'─'*50}")
    print(f"  USD Route Cost  : ₹{result['usd_route_cost']:>12,.2f}")
    print(f"  SNN Route Cost  : ₹{result['snn_route_cost']:>12,.2f}")
    print(f"  Saving/tx       : ₹{result['savings_inr']:>12,.2f}  "
          f"({result['savings_percentage']:.1f}%)")
    print(f"  Annual est.     : ₹{result['savings']['annual_estimate_inr']:>12,.2f}")
    print(f"  {'─'*50}")
    print(f"  Settlement now  : {result['settlement_time_current']}")
    print(f"  Settlement SNN  : {result['settlement_time_proposed']}")
    print(f"  {'─'*50}")
    print(f"  SNN prob        : {result['snn_prediction']['probability']:.4f}")
    print(f"  LIF spike rate  : {result['snn_prediction']['spike_rate_pct']:.2f}%")
    print(f"  Threshold used  : {result['snn_prediction']['prob_threshold_used']}")
    print(f"  API latency     : {latency_ms:.1f} ms")
    print(f"  {'='*50}\n")

    ok(f"direction={result['direction']}")
    ok(f"recommendation={result['recommendation']}")
    ok(f"savings=₹{result['savings_inr']:,.0f}  ({result['savings_percentage']:.1f}%)")
    ok(f"usd_route_cost > snn_route_cost ✓")
    ok(f"all required fields present ✓")
    ok(f"api latency={latency_ms:.0f}ms")
    return result


# ─────────────────────────────────────────────
# TEST 3: Bad input — wrong length
# ─────────────────────────────────────────────

def test_bad_input_wrong_length():
    """POST /predict with only 5 prices should return 400."""
    print(f"\n{BOLD}TEST 3 — Bad Input: Wrong Length{RESET}")

    payload = {
        "price_sequence"        : [16.2, 16.3, 16.1, 16.4, 16.5],
        "transaction_amount_inr": 1_000_000,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)

    assert resp.status_code in (400, 422), \
        f"Expected 400/422 for short sequence, got {resp.status_code}"

    ok(f"Returned {resp.status_code} for 5-price sequence (expected 400/422)")
    info(f"Error detail: {resp.json().get('detail', resp.text)[:100]}")


# ─────────────────────────────────────────────
# TEST 4: Bad input — negative price
# ─────────────────────────────────────────────

def test_bad_input_negative_price():
    """POST /predict with a negative price should return 400."""
    print(f"\n{BOLD}TEST 4 — Bad Input: Negative Price{RESET}")

    prices  = [16.2, 16.3, 16.1, 16.4, 16.5,
               16.3, 16.6, 16.4, 16.7, -1.0]
    payload = {"price_sequence": prices, "transaction_amount_inr": 1_000_000}
    resp    = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)

    assert resp.status_code in (400, 422), \
        f"Expected 400/422 for negative price, got {resp.status_code}"

    ok(f"Returned {resp.status_code} for negative price (expected 400/422)")
    info(f"Error detail: {resp.json().get('detail', resp.text)[:100]}")


# ─────────────────────────────────────────────
# TEST 5: Bad input — out of range prices
# ─────────────────────────────────────────────

def test_bad_input_out_of_range():
    """POST /predict with USD/INR rates (wrong pair) should return 400."""
    print(f"\n{BOLD}TEST 5 — Bad Input: Wrong Currency Pair{RESET}")

    # USD/INR rates (~83) instead of INR/BRL (~16) — should be rejected
    prices  = [83.2, 83.4, 83.1, 83.6, 83.5,
               83.3, 83.7, 83.4, 83.8, 83.5]
    payload = {"price_sequence": prices, "transaction_amount_inr": 1_000_000}
    resp    = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)

    assert resp.status_code in (400, 422), \
        f"Expected 400/422 for out-of-range prices, got {resp.status_code}"

    ok(f"Returned {resp.status_code} for USD/INR prices (wrong range)")


# ─────────────────────────────────────────────
# TEST 6: Summary endpoint
# ─────────────────────────────────────────────

def test_summary_endpoint():
    """GET /summary?amount=X should return cost comparison."""
    print(f"\n{BOLD}TEST 6 — Summary Endpoint{RESET}")

    amounts = [100_000, 1_000_000, 5_000_000]

    for amount in amounts:
        resp = requests.get(
            f"{BASE_URL}/summary",
            params={"amount": amount},
            timeout=10
        )
        assert resp.status_code == 200, \
            f"Expected 200 for amount={amount}, got {resp.status_code}"

        data = resp.json()
        assert data["usd_route_cost"] > data["snn_route_cost"], \
            f"SNN should be cheaper for amount={amount}"

        saving_pct = data["savings_percentage"]
        ok(f"amount=₹{amount:>10,}  saving={saving_pct:.1f}%  "
           f"₹{data['savings_inr']:,.0f}")


# ─────────────────────────────────────────────
# TEST 7: Three transaction sizes
# ─────────────────────────────────────────────

def test_three_transaction_sizes():
    """Test /predict with small, medium, large transaction amounts."""
    print(f"\n{BOLD}TEST 7 — Three Transaction Sizes{RESET}")

    prices = get_real_prices(10)
    sizes  = [
        (100_000,   "Small  (₹1 lakh) "),
        (1_000_000, "Medium (₹10 lakh)"),
        (5_000_000, "Large  (₹50 lakh)"),
    ]

    print(f"\n  {'Size':<22} {'USD Cost':>12}  {'SNN Cost':>10}  "
          f"{'Saving':>10}  {'%':>6}")
    print(f"  {'─'*65}")

    for amount, label in sizes:
        payload = {
            "price_sequence"        : prices,
            "transaction_amount_inr": amount,
            "monthly_tx_count"      : 5,
        }
        resp = requests.post(f"{BASE_URL}/predict",
                             json=payload, timeout=30)
        assert resp.status_code == 200, f"Failed for {label}"

        r = resp.json()
        print(f"  {label:<22} "
              f"₹{r['usd_route_cost']:>10,.0f}  "
              f"₹{r['snn_route_cost']:>8,.0f}  "
              f"₹{r['savings_inr']:>8,.0f}  "
              f"{r['savings_percentage']:>5.1f}%")

        assert r["savings_inr"] > 0, f"No saving for {label}"

    ok("All three transaction sizes returned valid savings")


# ─────────────────────────────────────────────
# TEST 8: Response time benchmark
# ─────────────────────────────────────────────

def test_response_time():
    """Measure /predict latency over 10 calls."""
    print(f"\n{BOLD}TEST 8 — Response Time Benchmark (10 calls){RESET}")

    prices  = get_real_prices(10)
    payload = {
        "price_sequence"        : prices,
        "transaction_amount_inr": 1_000_000,
    }
    latencies = []

    for i in range(10):
        t0   = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/predict",
                             json=payload, timeout=30)
        ms   = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        latencies.append(ms)

    mean_ms = np.mean(latencies)
    p95_ms  = np.percentile(latencies, 95)

    ok(f"Mean latency : {mean_ms:.1f} ms")
    ok(f"P95  latency : {p95_ms:.1f} ms")
    ok(f"Min/Max      : {min(latencies):.1f} / {max(latencies):.1f} ms")

    if mean_ms < 500:
        ok(f"Mean < 500ms — acceptable for demo ✓")
    else:
        info(f"Mean > 500ms — consider caching for dashboard")


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def run_all_tests():
    """Run all tests sequentially and report results."""
    tests = [
        ("Health Check",            test_health),
        ("Predict with Real Data",  test_predict_with_real_data),
        ("Bad Input: Wrong Length", test_bad_input_wrong_length),
        ("Bad Input: Negative",     test_bad_input_negative_price),
        ("Bad Input: Out of Range", test_bad_input_out_of_range),
        ("Summary Endpoint",        test_summary_endpoint),
        ("Three Transaction Sizes", test_three_transaction_sizes),
        ("Response Time",           test_response_time),
    ]

    passed, failed_tests = 0, []
    t_total = time.time()

    print(f"\n{BOLD}{'='*55}")
    print(f"  BRICS SNN API — END-TO-END TEST SUITE")
    print(f"  {BASE_URL}")
    print(f"{'='*55}{RESET}")

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            fail(f"FAILED: {name}")
            info(f"  Reason: {e}")
            failed_tests.append((name, str(e)))
        except requests.exceptions.ConnectionError:
            fail(f"FAILED: {name} — API not reachable at {BASE_URL}")
            fail("  → Is the API running? Start with:")
            fail("    uvicorn api.main:app --reload")
            failed_tests.append((name, "Connection refused"))
            break
        except Exception as e:
            fail(f"FAILED: {name} — {type(e).__name__}: {e}")
            failed_tests.append((name, str(e)))

    elapsed = time.time() - t_total

    # ── Final report ──────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*55}")
    print(f"  RESULTS: {passed}/{len(tests)} tests passed  "
          f"({elapsed:.1f}s)")
    print(f"{'='*55}{RESET}")

    if failed_tests:
        print(f"\n{RED}  Failed tests:{RESET}")
        for name, reason in failed_tests:
            print(f"    ❌ {name}: {reason}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}  ✅ All {passed} tests passed.{RESET}")
        print(f"  This output is your thesis proof-of-concept evidence.")
        print(f"  Screenshot this terminal for your appendix.\n")


if __name__ == "__main__":
    run_all_tests()