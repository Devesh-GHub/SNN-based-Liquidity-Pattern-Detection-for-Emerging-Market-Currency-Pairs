"""
dashboard/utils/formatters.py
==============================
Number and string formatting helpers for the dashboard.
All monetary values displayed in INR with ₹ symbol.
"""


def fmt_inr(amount: float, decimals: int = 0) -> str:
    """
    Format a float as an INR currency string with ₹ symbol
    and Indian-style comma grouping.

    Examples
    --------
    >>> fmt_inr(1000000)
    '₹10,00,000'
    >>> fmt_inr(34944.5, decimals=2)
    '₹34,944.50'
    """
    if decimals == 0:
        return f"₹{amount:,.0f}"
    return f"₹{amount:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.

    Parameters
    ----------
    value    : float — e.g. 96.58 (already in %)
    decimals : int   — decimal places

    Examples
    --------
    >>> fmt_pct(96.58)
    '96.58%'
    >>> fmt_pct(0.478, decimals=1)
    '0.5%'
    """
    return f"{value:.{decimals}f}%"


def fmt_confidence(confidence: float) -> str:
    """
    Convert raw confidence [0,1] to a human-readable label.

    Examples
    --------
    >>> fmt_confidence(0.72)
    'High (72%)'
    >>> fmt_confidence(0.24)
    'Low (24%)'
    """
    pct = confidence * 100
    if pct >= 60:
        label = "High"
    elif pct >= 30:
        label = "Moderate"
    else:
        label = "Low"
    return f"{label} ({pct:.0f}%)"


def fmt_settlement(days: str, hours: float) -> str:
    """
    Format settlement time as a combined string.

    Examples
    --------
    >>> fmt_settlement("T+2 to T+3", 48)
    'T+2 to T+3 (≈48 hours)'
    >>> fmt_settlement("T+0", 0.08)
    'T+0 (≈5 minutes)'
    """
    if hours < 1:
        time_str = f"≈{hours * 60:.0f} minutes"
    elif hours < 24:
        time_str = f"≈{hours:.0f} hours"
    else:
        time_str = f"≈{hours/24:.0f} days"
    return f"{days} ({time_str})"


def recommendation_emoji(rec: str) -> str:
    """Return emoji for recommendation string."""
    return "✅ DIRECT" if rec == "DIRECT" else "⚠️ USD FALLBACK"


def direction_emoji(direction: str) -> str:
    """Return emoji for direction string."""
    return "📈 UP" if direction == "UP" else "📉 DOWN"