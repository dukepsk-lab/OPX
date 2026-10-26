"""Alert-only decision rule and Telegram alert dispatcher."""

from dataclasses import dataclass
import httpx
from opx.econ import breakeven


@dataclass(frozen=True)
class Decision:
    side: str
    p_win: float
    stake: float
    k: int


def decide(
    p_call: float,
    p_put: float,
    r: float,
    buffer: float,
    equity: float,
    k: int,
    min_payout: float = 0.80,
    stake_frac: float = 0.01,
) -> Decision | None:
    """Evaluate whether to take a trade, which side, and calculate fixed stake sizing.
    
    Returns None if payout r < min_payout or highest win probability <= breakeven(r) + buffer.
    """
    if r < min_payout:
        return None

    if p_call >= p_put:
        side = "call"
        p_win = p_call
    else:
        side = "put"
        p_win = p_put

    threshold = breakeven(r) + buffer
    if p_win <= threshold:
        return None

    stake = round(equity * stake_frac, 2)
    return Decision(side=side, p_win=float(p_win), stake=float(stake), k=k)


def send_telegram(
    token: str,
    chat_id: str,
    text: str,
    client: httpx.Client | None = None,
) -> None:
    """Send an alert message to a Telegram chat via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    if client is not None:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
    else:
        with httpx.Client() as c:
            resp = c.post(url, json=payload)
            resp.raise_for_status()
