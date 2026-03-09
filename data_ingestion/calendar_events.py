from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import requests
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

def get_macro_calendar():
    """Free macro/geopolitical pulse from Google News RSS."""
    url = "https://news.google.com/rss/search?q=US+market+fed+inflation+war+tariff+oil&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "bloodbath-engine/1.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        headlines = []
        for item in items[:8]:
            title = (item.findtext("title") or "").strip()
            if title:
                headlines.append(title)

        txt = " ".join(headlines).lower()
        risk_terms = {
            "war": "geo_war",
            "sanction": "geo_sanctions",
            "tariff": "geo_tariff",
            "oil": "macro_oil",
            "inflation": "macro_inflation",
            "cpi": "macro_cpi",
            "fed": "macro_fed",
            "rate hike": "macro_rate_hike",
            "recession": "macro_recession",
        }
        flags = sorted({tag for term, tag in risk_terms.items() if term in txt})
        risk_level = "high" if len(flags) >= 3 else ("medium" if len(flags) >= 1 else "low")
        return {
            "today_highlights": headlines[:5],
            "risk_flags": flags,
            "risk_level": risk_level,
        }
    except Exception:
        return {
            "today_highlights": [],
            "risk_flags": [],
            "risk_level": "unknown",
        }

def get_earnings_calendar(symbols):
    """Free earnings context using yfinance calendar when available."""
    out = {}
    now = datetime.now(timezone.utc)
    if yf is None:
        for sym in symbols:
            out[str(sym).upper()] = {
                "status": "unavailable",
                "days_to_earnings": None,
                "earnings_upcoming_7d": False,
            }
        return out
    for sym in symbols:
        sym_u = str(sym).upper()
        default = {
            "status": "unknown",
            "days_to_earnings": None,
            "earnings_upcoming_7d": False,
        }
        try:
            cal = yf.Ticker(sym_u).calendar
            if cal is None or getattr(cal, "empty", False):
                out[sym_u] = {**default, "status": "unavailable"}
                continue

            idx = [str(x).lower() for x in getattr(cal, "index", [])]
            row = None
            for candidate in ("earnings date", "earnings"):
                if candidate in idx:
                    row = cal.loc[[x for x in cal.index if str(x).lower() == candidate][0]]
                    break
            if row is None:
                out[sym_u] = {**default, "status": "unavailable"}
                continue

            val = row.iloc[0] if hasattr(row, "iloc") else None
            if val is None:
                out[sym_u] = {**default, "status": "unavailable"}
                continue

            if hasattr(val, "to_pydatetime"):
                dt = val.to_pydatetime()
            else:
                dt = datetime.fromisoformat(str(val))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dte = (dt - now).total_seconds() / 86400.0
            out[sym_u] = {
                "status": "ok",
                "earnings_date": dt.isoformat(),
                "days_to_earnings": round(dte, 2),
                "earnings_upcoming_7d": 0 <= dte <= 7,
            }
        except Exception:
            out[sym_u] = {**default, "status": "unavailable"}
    return out
