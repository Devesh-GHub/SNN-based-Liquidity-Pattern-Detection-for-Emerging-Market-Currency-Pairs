"""
dashboard/utils/api_client.py
==============================
HTTP client for the BRICS SNN Settlement API.

All API calls go through this module. Handles timeouts, cold-start
warnings, and graceful fallback to demo data.

API URL is read from environment variable BRICS_API_URL, then from
Streamlit secrets, then falls back to localhost for local development.
"""

import requests
import streamlit as st
import os
from dashboard.config import PLACEHOLDER_DATA, API_TIMEOUT_S


# ── Resolve API URL ────────────────────────────────────────────────────
def _get_api_url() -> str:
    """
    Resolve API URL in priority order:
    1. Environment variable BRICS_API_URL
    2. Streamlit secrets (for Streamlit Cloud deployment)
    3. localhost fallback (local development)
    """
    # Environment variable (Render, Docker, local .env)
    env_url = os.environ.get("BRICS_API_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    # Streamlit secrets (Streamlit Cloud)
    try:
        secret_url = st.secrets.get("API_URL", "").strip()
        if secret_url:
            return secret_url.rstrip("/")
    except Exception:
        pass

    # Local fallback
    return "http://localhost:8000"


API_URL = _get_api_url()


# ── Core API call ──────────────────────────────────────────────────────

def call_predict_api(price_sequence     : list,
                     transaction_amount : float,
                     monthly_tx_count   : int = 5) -> tuple:
    """
    POST to /predict and return (result_dict, error_string).

    Returns (data, None) on success.
    Returns (None, error_message) on any failure.

    Parameters
    ----------
    price_sequence     : list of 10 floats — INR/BRL closing prices
    transaction_amount : float — INR value
    monthly_tx_count   : int   — for annual saving projection

    Returns
    -------
    tuple : (dict or None, str or None)

    Examples
    --------
    >>> data, err = call_predict_api([16.2]*10, 1_000_000)
    >>> if err: st.error(err)
    >>> else: print(data["recommendation"])
    """
    payload = {
        "price_sequence"        : [round(float(p), 4) for p in price_sequence],
        "transaction_amount_inr": float(transaction_amount),
        "monthly_tx_count"      : int(monthly_tx_count),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json    = payload,
            timeout = API_TIMEOUT_S,
            headers = {"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            return response.json(), None

        elif response.status_code == 400:
            detail = response.json().get("detail", "Bad request")
            return None, f"Input error: {detail}"

        elif response.status_code == 422:
            errors = response.json().get("detail", [])
            if isinstance(errors, list) and errors:
                msg = errors[0].get("msg", "Validation error")
            else:
                msg = "Validation error — check price sequence"
            return None, f"Validation: {msg}"

        elif response.status_code == 503:
            return None, (
                "API is warming up (Render free tier cold start). "
                "Please wait 30 seconds and try again."
            )

        else:
            return None, f"API error {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        return None, (
            "⏱️ Request timed out — the API may be waking up from sleep. "
            "Render free tier sleeps after 15 minutes of inactivity. "
            "Wait 30 seconds and try again."
        )
    except requests.exceptions.ConnectionError:
        return None, (
            f"Cannot connect to API at {API_URL}. "
            "If using local mode, start the API with: "
            "uvicorn api.main:app --reload"
        )
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def call_health_api() -> tuple:
    """
    GET /health and return (health_dict, error_string).

    Returns (dict, None) on success, (None, error) on failure.
    """
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Health check failed: {resp.status_code}"
    except requests.exceptions.Timeout:
        return None, "Health check timed out"
    except Exception as e:
        return None, str(e)


def call_summary_api(amount: float) -> tuple:
    """
    GET /summary?amount=X and return (summary_dict, error_string).
    Used for cost-only updates without inference.
    """
    try:
        resp = requests.get(
            f"{API_URL}/summary",
            params  = {"amount": float(amount)},
            timeout = 10,
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Summary error: {resp.status_code}"
    except Exception as e:
        return None, str(e)


def validate_price_input(raw_input: str) -> tuple:
    """
    Parse and validate a comma-separated price string.

    Returns (list_of_floats, None) on success.
    Returns (None, error_message) on failure.

    Parameters
    ----------
    raw_input : str — e.g. "16.2, 16.3, 16.1, ..."

    Examples
    --------
    >>> prices, err = validate_price_input("16.2, 16.3, 16.1")
    >>> # returns (None, "Need exactly 10 prices, got 3")
    """
    try:
        prices = [float(x.strip()) for x in raw_input.split(",")
                  if x.strip()]
    except ValueError:
        return None, "All values must be numbers (e.g. 16.2, 16.5, ...)"

    if len(prices) != 10:
        return None, (
            f"Need exactly 10 prices, got {len(prices)}. "
            "Enter the last 10 daily INR/BRL closing rates."
        )
    if any(p <= 0 for p in prices):
        return None, "All prices must be positive."
    if any(p > 100 or p < 5 for p in prices):
        return None, (
            "Prices must be in the INR/BRL range [5, 100]. "
            "USD/INR rates (~83) are not accepted here."
        )
    return prices, None