"""
api/models.py
=============
Pydantic request and response schemas for the BRICS SNN Settlement API.

Every endpoint input and output is typed here. Pydantic validates all
data automatically and generates the Swagger /docs UI for free.

Design principle
----------------
Schemas are deliberately flat (no nested required fields in requests)
to make the API easy to call from Streamlit, curl, and notebooks.
All monetary values are in INR. All percentages are in decimal form
(0.035 = 3.5%) internally but converted to human-readable % strings
in response fields named with "_pct" suffix.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────
# REQUEST SCHEMAS — what the client sends
# ─────────────────────────────────────────────


class PriceSignalRequest(BaseModel):
    """
    Input for the /predict endpoint.
    Send the last 10 daily INR/BRL closing rates and your transaction size.

    The client sends the last 10 daily closing INR/BRL rates and
    the transaction amount they want to route. The API handles all
    feature engineering and model inference internally.

    Fields
    ------
    price_sequence : list of 10 floats — last 10 INR/BRL closing rates
                     in chronological order (oldest first, newest last)
                     Example: [16.2, 16.3, 16.1, 16.4, 16.5,
                               16.3, 16.6, 16.4, 16.7, 16.5]

    transaction_amount_inr : float — INR value of the transaction
                             Default: ₹10,00,000 (10 lakh)
                             Min: ₹1,000  Max: ₹10,00,00,000 (10 crore)

    monthly_tx_count : int — transactions per month for annual projection
                       Default: 5

    Examples
    --------
    {
      "price_sequence": [16.2, 16.3, 16.1, 16.4, 16.5,
                         16.3, 16.6, 16.4, 16.7, 16.5],
      "transaction_amount_inr": 1000000.0,
      "monthly_tx_count": 5
    }
    """
    price_sequence : List[float] = Field(
        ...,
        description=(
            "Last 10 daily closing prices of the synthetic INR/BRL "
            "cross-rate, in chronological order (oldest first, newest last). "
            "Typical range: 15.0 – 18.0 INR per BRL."
        ),
        min_length = 10,
        max_length = 10,
        examples   = [[16.2, 16.3, 16.1, 16.4, 16.5,
                        16.3, 16.6, 16.4, 16.7, 16.5]],
    )
    transaction_amount_inr : float = Field(
        default     = 1_000_000.0,
        description = (
            "Transaction value in Indian Rupees (INR). "
            "Default: ₹10,00,000 (10 lakh). "
            "Range: ₹1,000 to ₹10,00,00,000."
        ),
        ge       = 1_000,        # minimum ₹1,000
        le       = 100_000_000,  # maximum ₹10 crore
        examples = [1_000_000.0],
    )
    monthly_tx_count : int = Field(
        default     = 5,
        description = (
            "Number of transactions per month — used to compute the "
            "annualised saving estimate in the response."
        ),
        ge       = 1,
        le       = 500,
        examples = [5],
    )

    @field_validator("price_sequence")
    @classmethod
    def validate_prices(cls, v):
        if len(v) != 10:
            raise ValueError(
                f"price_sequence must have exactly 10 values, got {len(v)}. "
                f"Provide the last 10 daily INR/BRL closing rates."
            )
        if any(p <= 0 for p in v):
            raise ValueError(
                "All prices must be positive."
            )

        # Broad sanity bounds.
        if any(p > 100 or p < 5 for p in v):
            raise ValueError(
                "Prices appear to be outside a broad sanity range [5, 100]. "
                "Verify you are sending INR/BRL rates."
            )

        # Currency-pair sanity check: INR/BRL synthetic series clusters around the mid-teens.
        # Reject sequences whose median is far outside that band (likely another FX pair).
        sorted_v = sorted(float(p) for p in v)
        median = (sorted_v[4] + sorted_v[5]) / 2.0
        if median < 10 or median > 30:
            raise ValueError(
                "Prices do not look like INR/BRL. Expected values roughly in the 10–30 band "
                "(synthetic INR/BRL is typically ~15–18). Verify you are not sending USD/INR (~83) "
                "or USD/BRL (~5)."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "price_sequence"        : [16.2, 16.3, 16.1, 16.4, 16.5,
                                           16.3, 16.6, 16.4, 16.7, 16.5],
                "transaction_amount_inr": 1_000_000.0,
                "monthly_tx_count"      : 5,
            }
        }
    }

class HealthRequest(BaseModel):
    """Optional body for the /health endpoint (for future auth)."""
    check_model : bool = Field(
        default=True,
        description="Whether to verify the SNN model is loaded"
    )


# ─────────────────────────────────────────────
# RESPONSE SCHEMAS — what the API returns
# ─────────────────────────────────────────────

class RouteDetail(BaseModel):
    """
    Cost breakdown for one settlement route (USD or SNN direct).
    Embedded in PredictionResponse.
    """
    total_cost_inr    : float = Field(description="Total cost in INR")
    cost_percentage   : float = Field(description="Cost as % of transaction")
    settlement_days   : str   = Field(description="e.g. 'T+2 to T+3' or 'T+0'")
    settlement_hours  : float = Field(description="Approximate hours to settle")
    steps             : List[str] = Field(
        description="Route steps e.g. ['INR→USD', 'USD→BRL']"
    )


class SavingsDetail(BaseModel):
    """Saving summary embedded in PredictionResponse."""
    amount_inr          : float = Field(description="Saving per transaction in INR")
    amount_formatted    : str   = Field(description="Formatted string e.g. '₹34,564.00'")
    percentage          : float = Field(description="Saving as % of USD route cost")
    latency_saving_days : float = Field(description="Days saved on settlement time")
    annual_estimate_inr : float = Field(
        description="Projected annual saving at monthly_tx_count rate"
    )
    annual_formatted    : str   = Field(description="Formatted annual saving string")


class SNNPredictionDetail(BaseModel):
    """SNN model output details embedded in PredictionResponse."""
    probability         : float = Field(description="Raw SNN output probability [0,1]")
    direction           : str   = Field(description="'UP' or 'DOWN'")
    confidence          : float = Field(description="Confidence score [0,1]")
    spike_rate          : float = Field(description="Mean LIF spike rate [0,1]")
    spike_rate_pct      : float = Field(description="Spike rate as percentage")
    recommendation      : str   = Field(description="'DIRECT' or 'USD_FALLBACK'")
    recommendation_text : str   = Field(description="Human-readable recommendation")
    prob_threshold_used : float = Field(description="Threshold used for decision")
    rate_threshold_used : float = Field(description="Spike rate threshold used")


class PredictionResponse(BaseModel):
    """
    Full response from the /predict endpoint.

    Contains SNN prediction, itemised costs for both routes,
    savings calculation, and a human-readable recommendation.
    All monetary values are in INR.

    This schema is consumed directly by the Streamlit dashboard
    and is the single source of truth for all displayed numbers.
    """
    # ── Top-level convenience fields (for quick dashboard display) ────
    direction                : str   = Field(
        description="Predicted INR/BRL direction: 'UP' or 'DOWN'"
    )
    confidence               : float = Field(
        description="Model confidence score in [0, 1]"
    )
    spike_rate               : float = Field(
        description="Mean SNN spike rate for this sequence"
    )
    recommendation           : str   = Field(
        description="Settlement recommendation: 'DIRECT' or 'USD_FALLBACK'"
    )

    # ── Cost details ──────────────────────────────────────────────────
    usd_route_cost           : float = Field(
        description="Total cost via USD route in INR"
    )
    snn_route_cost           : float = Field(
        description="Total cost via SNN direct route in INR"
    )
    savings_inr              : float = Field(
        description="Saving per transaction in INR"
    )
    savings_percentage       : float = Field(
        description="Saving as percentage of USD route cost"
    )
    settlement_time_current  : str   = Field(
        description="Current settlement time e.g. 'T+2 to T+3'"
    )
    settlement_time_proposed : str   = Field(
        description="Proposed settlement time e.g. 'T+0'"
    )

    # ── Nested detail objects ─────────────────────────────────────────
    usd_route       : RouteDetail       = Field(
        description="Full USD route breakdown"
    )
    snn_route       : RouteDetail       = Field(
        description="Full SNN direct route breakdown"
    )
    savings         : SavingsDetail     = Field(
        description="Full savings breakdown"
    )
    snn_prediction  : SNNPredictionDetail = Field(
        description="Full SNN model output"
    )

    # ── Metadata ──────────────────────────────────────────────────────
    transaction_amount_inr   : float = Field(
        description="Transaction value used for calculation"
    )
    model_version            : str   = Field(
        default="BRICSLiquiditySNN v1.0"
    )
    generated_at             : str   = Field(
        description="ISO timestamp of response generation"
    )
    assumptions              : List[str] = Field(
        description="Honest caveats about this response"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "direction"               : "UP",
                "confidence"              : 0.72,
                "spike_rate"              : 0.35,
                "recommendation"          : "DIRECT",
                "usd_route_cost"          : 36180.0,
                "snn_route_cost"          : 1236.0,
                "savings_inr"             : 34944.0,
                "savings_percentage"      : 96.58,
                "settlement_time_current" : "T+2 to T+3",
                "settlement_time_proposed": "T+0",
                "transaction_amount_inr"  : 1000000.0,
                "model_version"           : "BRICSLiquiditySNN v1.0",
                "generated_at"            : "2024-03-15T10:30:00",
                "assumptions"             : [
                    "0.1% direct fee assumed per BIS benchmark"
                ],
            }
        }
    }


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status          : str  = Field(description="'healthy' or 'degraded'")
    model_loaded    : bool = Field(description="True if SNN model is in memory")
    model_version   : str  = Field(description="Model identifier")
    val_auc         : float= Field(description="Model's validation AUC")
    uptime_seconds  : float= Field(description="API uptime in seconds")
    timestamp       : str  = Field(description="ISO timestamp")


class SummaryResponse(BaseModel):
    """
    Response from the /summary endpoint.
    Returns cost comparison for a given transaction amount
    without running model inference — uses latest cached prediction.
    """
    transaction_amount_inr   : float
    usd_route_cost           : float
    snn_route_cost           : float
    savings_inr              : float
    savings_percentage       : float
    annual_saving_estimate   : float
    settlement_time_current  : str
    settlement_time_proposed : str
    last_recommendation      : str
    generated_at             : str


class ErrorResponse(BaseModel):
    """Standard error response for all 4xx/5xx errors."""
    error   : str = Field(description="Error type")
    message : str = Field(description="Human-readable error message")
    detail  : Optional[str] = Field(
        default=None,
        description="Technical detail for debugging"
    )