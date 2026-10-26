import json
import httpx
import pytest
from opx.live import Decision, decide, send_telegram


def test_picks_better_side_and_fixed_stake():
    d = decide(0.58, 0.51, r=0.85, buffer=0.02, equity=1000.0, k=5)
    assert d == Decision("call", 0.58, 10.0, 5)


def test_below_threshold_returns_none():
    assert decide(0.55, 0.50, r=0.85, buffer=0.02, equity=1000.0, k=5) is None  # 0.55 < 0.5405 + 0.02


def test_payout_floor_vetoes():
    assert decide(0.70, 0.40, r=0.75, buffer=0.0, equity=1000.0, k=5) is None


def test_send_telegram_posts_expected_request():
    seen = []
    transport = httpx.MockTransport(lambda req: seen.append(req) or httpx.Response(200, json={"ok": True}))
    send_telegram("TOKEN", "42", "hi", client=httpx.Client(transport=transport))
    assert seen[0].url.path == "/botTOKEN/sendMessage"
    assert json.loads(seen[0].content) == {"chat_id": "42", "text": "hi"}


def test_send_telegram_raises_on_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        send_telegram("T", "1", "x", client=httpx.Client(transport=transport))
