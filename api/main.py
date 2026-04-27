"""
api/main.py
===========
FastAPI application — BRICS SNN Settlement API.

Run with:
    uvicorn api.main:app --reload --app-dir .

Endpoints:
    GET  /           — root info
    GET  /health     — model health check
    POST /predict    — SNN inference + cost comparison
    GET  /summary    — cost-only (no inference)
    GET  /docs       — Swagger UI (auto-generated)
    GET  /redoc      — ReDoc UI (auto-generated)
"""

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import time
import os
import sys


# ── Path setup ────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

from api.models import (
    PriceSignalRequest,
    PredictionResponse,
    HealthResponse,
    SummaryResponse,
    RouteDetail,
    SavingsDetail,
    SNNPredictionDetail,
)
from api.predictor   import predictor
from api.cost_engine import get_dashboard_summary

# ── Startup timestamp ─────────────────────────────────────────────────
_API_START_TIME = time.time()


# ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Runs at startup (before first request) and at shutdown.
    The predictor singleton is already instantiated at import time,
    so this lifespan just logs the state and verifies readiness.
    """
    # ── Startup ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  BRICS SNN Settlement API  —  Starting up")
    print("=" * 50)

    status_info = predictor.get_status()
    if status_info["loaded"]:
        print(f"  ✅ SNN model ready")
        print(f"     Parameters : {status_info['parameters']}")
        print(f"     Val AUC    : {status_info['val_auc']}")
        print(f"     Load time  : {status_info['load_time_s']}s")
    else:
        print("  ⚠️  SNN model NOT loaded — /predict will return 503")
        print("     Check that outputs/ contains snn_model_best.pth")

    print("=" * 50 + "\n")

    yield   # API runs here

    # ── Shutdown ──────────────────────────────────────────────────────
    print("\n🛑  BRICS SNN API shutting down")


tags_metadata = [
    {
        "name"       : "Info",
        "description": "API health and status endpoints.",
    },
    {
        "name"       : "Prediction",
        "description": (
            "**Core endpoint.** Run SNN inference on INR/BRL price sequence "
            "and receive settlement routing recommendation with cost comparison."
        ),
    },
    {
        "name"       : "Cost Analysis",
        "description": (
            "Cost comparison endpoints — no model inference. "
            "Use for the dashboard cost calculator slider."
        ),
    },
]


# ── App definition ────────────────────────────────────────────────────
app = FastAPI(
    title            = "BRICS SNN Liquidity API",
    description      = (
        "## Event-Driven Synthetic Liquidity Discovery\n"
        "### for BRICS INR/BRL Settlement\n\n"
        "This API serves a trained **Spiking Neural Network** (SNN) that predicts "
        "optimal settlement routing for INR/BRL foreign exchange transactions.\n\n"
        "### How it works\n"
        "1. Send the last 10 daily INR/BRL closing prices\n"
        "2. The SNN encodes price movements as **spike trains**\n"
        "3. LIF neurons accumulate spikes over the 10-day window\n"
        "4. The model outputs a direction prediction (UP/DOWN) and "
        "   confidence score\n"
        "5. If confidence exceeds threshold → route **DIRECT** (save ~97%)\n"
        "6. Otherwise → route via **USD SWIFT** (safe fallback)\n\n"
        "### Key research numbers\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Model | BRICSLiquiditySNN |\n"
        "| Val AUC | 0.555 (vs LSTM 0.514) |\n"
        "| Parameters | 2,945 (vs LSTM 5,537) |\n"
        "| Energy reduction | ~10.6× vs LSTM |\n"
        "| Fee saving | ~97% per transaction |\n"
        "| Settlement time | T+2 → T+0 |\n\n"
        "⚠️ *Research demonstration only. 0.1% direct settlement fee is assumed.*"
    ),
    version          = "1.0.0",
    openapi_tags     = tags_metadata,
    lifespan         = lifespan,
    docs_url         = "/docs",
    redoc_url        = "/redoc",
    contact          = {
        "name" : "BRICS SNN Research Project",
        "email": "your.email@institution.ac.in",
    },
    license_info     = {
        "name": "MIT",
    },
)


# ── CORS ──────────────────────────────────────────────────────────────
# Needed so the Streamlit dashboard (Month 3) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to dashboard URL in production
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _build_prediction_response(
        snn_result        : dict,
        cost_summary      : dict,
        transaction_amount: float) -> PredictionResponse:
    """
    Assemble a PredictionResponse from SNN output and cost engine output.

    Converts all values to Python floats before building the Pydantic
    model — avoids numpy scalar serialisation errors.

    Parameters
    ----------
    snn_result         : dict from predictor.predict()
    cost_summary       : dict from get_dashboard_summary()
    transaction_amount : float

    Returns
    -------
    PredictionResponse — fully populated Pydantic model
    """
    u  = cost_summary["usd_route"]
    s  = cost_summary["snn_route"]
    sv = cost_summary["savings"]
    p  = cost_summary["snn_prediction"]

    return PredictionResponse(
        # ── Top-level convenience fields ──────────────────────────────
        direction                = str(snn_result["direction"]),
        confidence               = float(snn_result["confidence"]),
        spike_rate               = float(snn_result["spike_rate"]),
        recommendation           = str(snn_result["recommendation"]),
        usd_route_cost           = float(u["total_cost_inr"]),
        snn_route_cost           = float(s["total_cost_inr"]),
        savings_inr              = float(sv["amount_inr"]),
        savings_percentage       = float(sv["percentage"]),
        settlement_time_current  = str(u["settlement_days"]),
        settlement_time_proposed = str(s["settlement_days"]),

        # ── Nested: USD route ─────────────────────────────────────────
        usd_route = RouteDetail(
            total_cost_inr   = float(u["total_cost_inr"]),
            cost_percentage  = float(u["cost_percentage"]),
            settlement_days  = str(u["settlement_days"]),
            settlement_hours = float(u["settlement_hours"]),
            steps            = list(u["steps"]),
        ),

        # ── Nested: SNN route ─────────────────────────────────────────
        snn_route = RouteDetail(
            total_cost_inr   = float(s["total_cost_inr"]),
            cost_percentage  = float(s["cost_percentage"]),
            settlement_days  = str(s["settlement_days"]),
            settlement_hours = float(s["settlement_hours"]),
            steps            = list(s["steps"]),
        ),

        # ── Nested: savings ───────────────────────────────────────────
        savings = SavingsDetail(
            amount_inr          = float(sv["amount_inr"]),
            amount_formatted    = str(sv["amount_formatted"]),
            percentage          = float(sv["percentage"]),
            latency_saving_days = float(sv["latency_saving_days"]),
            annual_estimate_inr = float(sv["annual_estimate_inr"]),
            annual_formatted    = str(sv["annual_formatted"]),
        ),

        # ── Nested: SNN prediction detail ────────────────────────────
        snn_prediction = SNNPredictionDetail(
            probability         = float(snn_result["prob"]),
            direction           = str(snn_result["direction"]),
            confidence          = float(snn_result["confidence"]),
            spike_rate          = float(snn_result["spike_rate"]),
            spike_rate_pct      = float(snn_result["spike_rate_pct"]),
            recommendation      = str(snn_result["recommendation"]),
            recommendation_text = str(snn_result["recommendation_text"]),
            prob_threshold_used = float(snn_result["prob_threshold_used"]),
            rate_threshold_used = float(snn_result["rate_threshold_used"]),
        ),

        # ── Metadata ─────────────────────────────────────────────────
        transaction_amount_inr = float(transaction_amount),
        model_version          = "BRICSLiquiditySNN v1.0",
        generated_at           = datetime.utcnow().isoformat() + "Z",
        assumptions            = list(cost_summary.get("assumptions", [])),
    )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/",
         tags=["Info"],
         summary="API root")
def root():
    """
    Root endpoint. Confirms the API is running and lists all endpoints.
    """
    return {
        "name"         : "BRICS SNN Liquidity API",
        "version"      : "1.0.0",
        "status"       : "running",
        "model_loaded" : predictor.loaded,
        "uptime_s"     : round(time.time() - _API_START_TIME, 1),
        "endpoints"    : {
            "GET  /"        : "This info",
            "GET  /health"  : "Model health check",
            "POST /predict" : "SNN inference + cost comparison",
            "GET  /summary" : "Cost comparison (no inference)",
            "GET  /docs"    : "Swagger UI",
            "GET  /redoc"   : "ReDoc UI",
        },
    }


@app.get("/health",
         response_model = HealthResponse,
         tags           = ["Info"],
         summary        = "Model health check")
def health_check():
    """
    Health check endpoint.

    Returns 200 if model is loaded and ready.
    Returns 503 if model failed to load at startup.

    Called by the Streamlit dashboard every 30 seconds.
    """
    info   = predictor.get_status()
    uptime = round(time.time() - _API_START_TIME, 1)

    if not info["loaded"]:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "SNN model not loaded. "
                "Ensure outputs/snn_model.pth exists."
            )
        )

    return HealthResponse(
        status         = "healthy",
        model_loaded   = True,
        model_version  = info["model_version"],
        val_auc        = float(info["val_auc"]),
        uptime_seconds = uptime,
        timestamp      = datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict",
          response_model = PredictionResponse,
          tags           = ["Prediction"],
          summary        = "SNN inference + settlement cost comparison",
          responses      = {
              400: {"description": "Invalid price sequence"},
              422: {"description": "Validation error"},
              503: {"description": "Model not loaded"},
              500: {"description": "Internal inference error"},
          })
def predict(request: PriceSignalRequest):
    """
    **Core prediction endpoint.**

    Accepts the last 10 INR/BRL daily closing rates and a transaction
    amount. Returns the SNN's settlement recommendation plus a full
    cost breakdown comparing the USD route and SNN direct route.

    ### Input
```json
    {
      "price_sequence": [16.2, 16.3, 16.1, 16.4, 16.5,
                         16.3, 16.6, 16.4, 16.7, 16.5],
      "transaction_amount_inr": 1000000.0,
      "monthly_tx_count": 5
    }
```

    ### Output
    - `recommendation`: **"DIRECT"** or **"USD_FALLBACK"**
    - `savings_inr`: estimated saving per transaction
    - `spike_rate`: SNN mean spike rate (efficiency indicator)
    - Full itemised cost breakdown for both routes

    ### Decision logic
    DIRECT is recommended when:
    1. SNN probability ≥ 0.70 (favourable rate direction predicted)
    2. Spike rate ≥ 0.10 (sufficient signal richness)

    Otherwise falls back to conventional USD SWIFT route.
    """
    # ── Guard: model must be loaded ───────────────────────────────────
    if not predictor.loaded:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "SNN model is not loaded. "
                "The API is starting up or encountered an error. "
                "Try again in a moment."
            )
        )

    # ── Validate price sequence ───────────────────────────────────────
    if len(request.price_sequence) != 10:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = (
                f"price_sequence must have exactly 10 values, "
                f"got {len(request.price_sequence)}. "
                f"Provide the last 10 daily INR/BRL closing rates."
            )
        )

    if any(p <= 0 for p in request.price_sequence):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "All prices in price_sequence must be positive."
        )

    # ── Run SNN inference ─────────────────────────────────────────────
    try:
        snn_result = predictor.predict(request.price_sequence)
    except ValueError as e:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"SNN inference failed: {str(e)}"
        )

    # ── Get cost comparison ───────────────────────────────────────────
    try:
        cost_summary = get_dashboard_summary(
            transaction_amount = float(request.transaction_amount_inr),
            prediction         = snn_result,
            monthly_tx_count   = int(request.monthly_tx_count),
        )
    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Cost engine failed: {str(e)}"
        )

    # ── Assemble and return response ──────────────────────────────────
    try:
        response = _build_prediction_response(
            snn_result         = snn_result,
            cost_summary       = cost_summary,
            transaction_amount = request.transaction_amount_inr,
        )
    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Response assembly failed: {str(e)}"
        )

    return response


@app.get("/summary",
         response_model = SummaryResponse,
         tags           = ["Cost Analysis"],
         summary        = "Cost comparison without inference")
def summary(
    amount : float = Query(
        default     = 1_000_000,
        description = "Transaction value in INR",
        ge          = 1_000,
        le          = 100_000_000,
    )
):
    """
    Returns a cost comparison for a given transaction amount using
    the latest cached SNN prediction. Does **not** run new model
    inference — suitable for the dashboard's cost calculator slider.

    Faster than `/predict` because it skips the forward pass.
    """
    try:
        data = get_dashboard_summary(transaction_amount=float(amount))
        u    = data["usd_route"]
        s    = data["snn_route"]
        sv   = data["savings"]
        p    = data["snn_prediction"]

        return SummaryResponse(
            transaction_amount_inr  = float(amount),
            usd_route_cost          = float(u["total_cost_inr"]),
            snn_route_cost          = float(s["total_cost_inr"]),
            savings_inr             = float(sv["amount_inr"]),
            savings_percentage      = float(sv["percentage"]),
            annual_saving_estimate  = float(sv["annual_estimate_inr"]),
            settlement_time_current = str(u["settlement_days"]),
            settlement_time_proposed= str(s["settlement_days"]),
            last_recommendation     = str(p["recommendation"]),
            generated_at            = data["meta"]["generated_at"],
        )
    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Summary failed: {str(e)}"
        )


# ── Global exception handler ──────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all for any unhandled exceptions."""
    return JSONResponse(
        status_code = 500,
        content     = {
            "error"  : "InternalServerError",
            "message": "An unexpected error occurred",
            "detail" : str(exc),
        }
    )