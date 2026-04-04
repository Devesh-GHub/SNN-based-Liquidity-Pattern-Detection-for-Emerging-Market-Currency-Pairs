"""
src/fastapi_hello.py
====================
FastAPI hello-world for the BRICS SNN project.

This is a LEARNING file only — not the real prediction API.
The real API (with /predict and /health endpoints) will be built
in Month 2 once the SNN model is trained.

To run:
    uvicorn src.fastapi_hello:app --reload

Then open:
    http://localhost:8000/hello        ← your first endpoint
    http://localhost:8000/docs         ← auto-generated Swagger UI
    http://localhost:8000/redoc        ← alternative docs UI
"""

from fastapi import FastAPI
from pydantic import BaseModel   # data validation and parsing (validates and structures request/response data)
from datetime import datetime
from typing import Optional    # optional --> makes certain fields flexible (not mandatory)
import sys  
import os

# ── App initialisation ────────────────────────────────────────────────
app = FastAPI(
    title       = "BRICS SNN Settlement API",
    description = (
        "Event-Driven Synthetic Liquidity Discovery for BRICS Settlements "
        "using Spiking Neural Networks.\n\n"
        "**Status:** Learning / Hello-World phase.\n"
        "Real prediction endpoints arrive in Month 2."
    ),
    version     = "0.0.1",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── ENDPOINT 1: Root ──────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Root endpoint — confirms the API server is running.
    """
    return {
        "project" : "BRICS SNN FX Settlement",
        "status"  : "alive",
        "docs"    : "/docs",
        "version" : "0.0.1",
    }


# ── ENDPOINT 2: Hello ─────────────────────────────────────────────────

@app.get("/hello")
def hello():
    """
    Hello-world endpoint.

    Returns a confirmation message with a server timestamp.
    This is the first endpoint you test to confirm the server
    is reachable and responding correctly.
    """
    return {
        "message"   : "BRICS SNN API is alive",
        "timestamp" : datetime.utcnow().isoformat() + "Z",
        "author"    : "BRICS SNN Project",
    }


# ── ENDPOINT 3: Health ────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Health check endpoint.

    In production APIs, /health is called by load balancers and
    monitoring tools every few seconds to confirm the service is up.
    In Month 2, this will also report whether the SNN model is loaded.
    Returns HTTP 200 if healthy.
    """
    return {
        "status"       : "healthy",
        "model_loaded" : False,        # will be True in Month 2
        "model_version": None,         # will be "snn_v1" in Month 2
        "timestamp"    : datetime.utcnow().isoformat() + "Z",
    }


# ── ENDPOINT 4: API info (preview of Month 2 shape) ───────────────────

@app.get("/info")
def info():
    """
    API information and Month 2 endpoint preview.

    Shows what the full API will look like once the SNN model
    is integrated. Use this to plan your dashboard integration.
    """
    return {
        "current_endpoints": [
            {"path": "/",       "method": "GET", "status": "live"},
            {"path": "/hello",  "method": "GET", "status": "live"},
            {"path": "/health", "method": "GET", "status": "live"},
            {"path": "/info",   "method": "GET", "status": "live"},
        ],
        "month2_endpoints": [
            {
                "path"       : "/predict",
                "method"     : "POST",
                "status"     : "planned",
                "description": "Accept 10-day price window, return spike prediction + confidence",
                "input_shape": "(10, 9) — 10 timesteps × 9 features",
                "output"     : {
                    "direction"         : "1 (up) or 0 (down)",
                    "confidence"        : "float in [0, 1]",
                    "spike_rate"        : "fraction of input timesteps that spiked",
                    "settlement_signal" : "SETTLE_NOW or WAIT",
                },
            },
            {
                "path"       : "/encode",
                "method"     : "POST",
                "status"     : "planned",
                "description": "Accept raw price array, return spike-encoded binary sequence",
            },
        ],
    }


# ── Pydantic model preview (for /predict in Month 2) ──────────────────

class PredictRequest(BaseModel):
    """
    Input schema for the /predict endpoint (Month 2).

    Not active yet — defined here so you can see it in /docs
    and understand what JSON the dashboard will send.
    """
    price_window : list[float]       # 10 recent closing prices
    currency_pair: str = "INR/BRL"   # default pair
    threshold    : float = 0.003     # spike encoding threshold


class PredictResponse(BaseModel):
    """
    Output schema for the /predict endpoint (Month 2).
    """
    direction        : int            # 1=up, 0=down
    confidence       : float          # model output probability
    spike_rate       : float          # fraction of spiking timesteps
    settlement_signal: str            # "SETTLE_NOW" or "WAIT"
    latency_ms       : Optional[float] = None  # inference time


# Preview endpoint — shows the schema but returns placeholder data
@app.post("/predict/preview", response_model=PredictResponse)
def predict_preview(request: PredictRequest):
    """
    **PLACEHOLDER ONLY** — shows the /predict contract for Month 2.

    Accepts the real input format and returns dummy output.
    Replace this with actual SNN inference in Month 2.
    """
    if len(request.price_window) != 10:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"price_window must have exactly 10 values, "
                   f"got {len(request.price_window)}"
        )

    return PredictResponse(
        direction         = 1,              # placeholder
        confidence        = 0.5,            # placeholder
        spike_rate        = 0.144,          # placeholder — our calibrated rate
        settlement_signal = "PLACEHOLDER — model not yet trained",
        latency_ms        = 0.0,
    )


# ── Run directly (alternative to uvicorn CLI) ─────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.fastapi_hello:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        log_level="info",
    )