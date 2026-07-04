import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _write_output(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def main() -> int:
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    api_secret = os.environ.get("ALPACA_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print("Missing Alpaca credentials for market clock check.", file=sys.stderr)
        return 1

    request = Request(
        "https://paper-api.alpaca.markets/v2/clock",
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "Accept": "application/json",
            "User-Agent": "bloodbath-market-check/1.0",
        },
    )

    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        print(f"Failed to query Alpaca clock: HTTP {exc.code}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Failed to query Alpaca clock: {exc}", file=sys.stderr)
        return 1

    market_open = bool(payload.get("is_open", False))
    _write_output("market_open", "true" if market_open else "false")
    print(f"Alpaca market open: {market_open}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
