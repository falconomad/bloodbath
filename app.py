import streamlit as st
import pandas as pd
import plotly.express as px
import json
import base64
from pathlib import Path

from src.db import get_connection, init_db
from src.settings import TOP20_STARTING_CAPITAL
from src.common.trace_utils import load_jsonl_dict_rows
from src.analytics.explainability_report import generate_explainability_report
from src.ui.stock_logos import get_logo_url

# Polling refresh support. Prefer component if available, otherwise fall back to meta refresh.
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    st_autorefresh = None

st.set_page_config(
    page_title="kaibot",
    page_icon="assets/kaibot_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

logo_path = Path("assets/kaibot_logo.svg")
logo_uri = ""
if logo_path.exists():
    logo_uri = f"data:image/svg+xml;base64,{base64.b64encode(logo_path.read_bytes()).decode('utf-8')}"


theme_base = st.get_option("theme.base") or "light"
is_dark = theme_base.lower() == "dark"

if is_dark:
    bg = "#111317"
    card = "#171a20"
    border = "#282d38"
    text = "#e8edf4"
    panel = "#141821"
    muted_text = "#9ca7bb"
    accent = "#4d8dff"
    up_color = "#2ecc71"
    down_color = "#ff6b6b"
    neutral_color = "#9ca7bb"
    accent_soft = "rgba(77, 141, 255, 0.15)"
    metric_bg_1 = "#1b2333"
    metric_bg_2 = "#1b2a2a"
    metric_bg_3 = "#232033"
    plot_template = "plotly_dark"
    pie_palette = [
        "#3468BF",
        "#2B8F8F",
        "#4F9768",
        "#B38B3A",
        "#C06E56",
        "#8E63BC",
        "#3F9D93",
        "#739E32",
        "#B45A12",
        "#B8334A",
    ]
else:
    bg = "#f3f4f6"
    card = "#ffffff"
    border = "#e3e6ec"
    text = "#141821"
    panel = "#ffffff"
    muted_text = "#6f7684"
    accent = "#2f6df6"
    up_color = "#1b9e4b"
    down_color = "#d93025"
    neutral_color = "#6f7684"
    accent_soft = "rgba(47, 109, 246, 0.08)"
    metric_bg_1 = "#f7f9ff"
    metric_bg_2 = "#f5faf8"
    metric_bg_3 = "#f8f7ff"
    plot_template = "plotly_white"
    pie_palette = [
        "#2F6DF6",
        "#10B981",
        "#F59E0B",
        "#EF4444",
        "#8B5CF6",
        "#06B6D4",
        "#84CC16",
        "#F97316",
        "#EC4899",
        "#6366F1",
    ]

st.markdown(
    f"""
    <style>
      html, body, [data-testid="stAppViewContainer"], .stApp {{
        background: {bg} !important;
        color: {text};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
      }}
      [data-testid="stAppViewContainer"] > .main {{
        background: {bg} !important;
      }}
      [data-testid="stSidebar"] {{
        background: {panel} !important;
        border-right: 1px solid {border};
      }}
      [data-testid="stHeader"] {{
        background: {bg};
        border-bottom: 1px solid {border};
        height: 4.75rem;
        min-height: 4.75rem;
        position: fixed;
        top: 0;
        z-index: 1000;
        backdrop-filter: blur(10px);
      }}
      .kb-sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.1rem 0 0.8rem 0;
        position: relative;
      }}
      .kb-sidebar-brand img {{
        width: 4.3rem;
        height: 4.3rem;
        object-fit: contain;
      }}
      .kb-sidebar-brand span {{
        font-size: 2.28rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: {text};
        line-height: 1;
        position: absolute;
        left: 62px;
      }}
      @media (min-width: 961px) {{
        [data-testid="stHeader"] {{
          display: none !important;
        }}
        .block-container {{
          padding-top: 1.4rem !important;
        }}
      }}
      @media (max-width: 960px) {{
        .kb-sidebar-brand {{
          display: none;
        }}
        [data-testid="stHeader"] {{
          display: block !important;
        }}
        [data-testid="stHeader"]::before {{
          content: "";
          position: absolute;
          left: 0.8rem;
          top: 50%;
          width: 2.15rem;
          height: 2.15rem;
          transform: translateY(-50%);
          border-radius: 0.4rem;
          background-image: url('{logo_uri}');
          background-size: contain;
          background-repeat: no-repeat;
          background-position: center;
        }}
        [data-testid="stHeader"]::after {{
          content: "kaibot";
          position: absolute;
          left: 2.7rem;
          top: 50%;
          transform: translateY(-50%);
          font-size: 1.25rem;
          font-weight: 700;
          letter-spacing: -0.01em;
          color: {text};
          line-height: 1;
        }}
      }}
      [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{
        visibility: hidden;
        height: 0;
        position: fixed;
      }}
      .block-container {{
        padding-top: 7.3rem;
        max-width: 1200px;
      }}
      [data-testid="stMetricLabel"], .stMarkdown p {{
        color: {muted_text};
      }}
      [data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(18, 24, 33, 0.04);
        padding: 12px 16px;
        animation: bb-fade-rise 0.65s ease both;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
      }}
      [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(18, 24, 33, 0.09);
      }}
      /* Portfolio metric cards (first row, 3 columns) */
      [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] {{
        background: {metric_bg_1};
        animation: bb-fade-rise 0.65s ease both, bb-bg-fade 5.0s ease-in-out 0.25s infinite alternate;
      }}
      [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {{
        background: {metric_bg_2};
        animation: bb-fade-rise 0.65s ease both, bb-bg-fade 5.0s ease-in-out 0.5s infinite alternate;
      }}
      [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {{
        background: {metric_bg_3};
        animation: bb-fade-rise 0.65s ease both, bb-bg-fade 5.0s ease-in-out 0.75s infinite alternate;
      }}
      [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stExpander"] {{
        animation: bb-fade-rise 0.82s ease both;
        background: {card};
        border-radius: 16px;
      }}
      [data-testid="stPlotlyChart"] > div,
      [data-testid="stDataFrame"] > div,
      [data-testid="stExpander"] {{
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(18, 24, 33, 0.04);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
      }}
      [data-testid="stDataFrame"] > div,
      [data-testid="stExpander"] {{
        border: 1px solid {border};
      }}
      [data-testid="stPlotlyChart"]:hover > div,
      [data-testid="stDataFrame"]:hover > div,
      [data-testid="stExpander"]:hover {{
        box-shadow: 0 14px 28px rgba(18, 24, 33, 0.08);
      }}
      .stDataFrame [data-testid="stDataFrameResizable"] {{
        border-radius: 16px;
      }}
      [data-testid="stDataFrame"] img {{
        width: 14px !important;
        height: 14px !important;
        object-fit: contain;
      }}
      div[data-testid="stExpander"] details summary p {{
        color: {text} !important;
        font-weight: 500;
      }}
      [data-baseweb="select"] > div,
      .stTextInput > div > div,
      .stNumberInput > div > div {{
        border-radius: 12px;
        border-color: {border};
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }}
      [data-baseweb="select"] > div:hover,
      .stTextInput > div > div:hover,
      .stNumberInput > div > div:hover {{
        border-color: {border};
      }}
      [data-baseweb="select"] > div:focus-within,
      .stTextInput > div > div:focus-within,
      .stNumberInput > div > div:focus-within {{
        border-color: {border};
        box-shadow: none;
      }}
      [data-testid="stPlotlyChart"] {{
        animation-delay: 0.15s;
      }}
      @keyframes bb-fade-rise {{
        from {{
          opacity: 0;
          transform: translateY(12px);
          filter: blur(2px);
        }}
        to {{
          opacity: 1;
          transform: translateY(0);
          filter: blur(0);
        }}
      }}
      @keyframes bb-bg-fade {{
        from {{
          filter: saturate(1) brightness(1);
        }}
        to {{
          filter: saturate(1.06) brightness(0.97);
        }}
      }}
      h2, h3 {{
        letter-spacing: -0.02em;
        font-weight: 600;
      }}
      [data-testid="stSkeleton"] {{
        display: none !important;
      }}
      .bb-skeleton-wrap {{
        display: grid;
        gap: 1rem;
      }}
      .bb-skeleton-row {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
      }}
      .bb-skeleton-chart-row {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
      }}
      .bb-skeleton-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(18, 24, 33, 0.04);
        overflow: hidden;
      }}
      .bb-skeleton-metric {{
        height: 108px;
      }}
      .bb-skeleton-chart {{
        height: 360px;
      }}
      .bb-skeleton-table {{
        height: 320px;
      }}
      .bb-skeleton-pulse {{
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.16), transparent);
        background-size: 250% 100%;
        animation: bb-skeleton-shimmer 1.4s ease infinite;
      }}
      @keyframes bb-skeleton-shimmer {{
        0% {{ background-position: 100% 0; }}
        100% {{ background-position: -100% 0; }}
      }}
      @media (max-width: 900px) {{
        .bb-skeleton-row, .bb-skeleton-chart-row {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def show_dashboard_skeleton():
    st.markdown(
        """
        <div class="bb-skeleton-wrap">
          <div class="bb-skeleton-row">
            <div class="bb-skeleton-card bb-skeleton-metric"><div class="bb-skeleton-pulse"></div></div>
            <div class="bb-skeleton-card bb-skeleton-metric"><div class="bb-skeleton-pulse"></div></div>
            <div class="bb-skeleton-card bb-skeleton-metric"><div class="bb-skeleton-pulse"></div></div>
          </div>
          <div class="bb-skeleton-card bb-skeleton-chart"><div class="bb-skeleton-pulse"></div></div>
          <div class="bb-skeleton-chart-row">
            <div class="bb-skeleton-card bb-skeleton-chart"><div class="bb-skeleton-pulse"></div></div>
            <div class="bb-skeleton-card bb-skeleton-chart"><div class="bb-skeleton-pulse"></div></div>
          </div>
          <div class="bb-skeleton-card bb-skeleton-table"><div class="bb-skeleton-pulse"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _color_signed(v):
    try:
        n = float(v)
    except Exception:
        return ""
    if n > 0:
        return f"color: {up_color}; font-weight: 600;"
    if n < 0:
        return f"color: {down_color}; font-weight: 600;"
    return f"color: {neutral_color};"


def _color_decision(v):
    value = str(v).upper()
    if value == "BUY":
        return f"color: {up_color}; font-weight: 700;"
    if value == "SELL":
        return f"color: {down_color}; font-weight: 700;"
    return f"color: {neutral_color}; font-weight: 600;"


def _signed_bar_hex(v, max_abs):
    try:
        n = float(v)
    except Exception:
        return neutral_color
    if max_abs <= 0:
        return neutral_color

    strength = min(abs(n) / max_abs, 1.0)
    if n >= 0:
        # Higher gains -> darker green.
        return "#1E7A46" if strength < 0.35 else ("#165C35" if strength < 0.7 else "#0F3D24")
    # Higher losses -> darker red.
    return "#A14A4A" if strength < 0.35 else ("#7A3434" if strength < 0.7 else "#4D1F1F")


def _load_latest_experiment_result():
    root = Path("artifacts/experiments")
    if not root.exists():
        return None


def _with_logo_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ticker" not in df.columns:
        return df
    out = df.copy()
    out["logo"] = out["ticker"].astype(str).map(get_logo_url)
    return out
    json_files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_files:
        return None
    try:
        return json.loads(json_files[0].read_text(encoding="utf-8")), json_files[0]
    except Exception:
        return None


skeleton_placeholder = st.empty()
with skeleton_placeholder.container():
    show_dashboard_skeleton()

db_ready = True
try:
    init_db()
except Exception as exc:
    db_ready = False
    st.warning(f"Database unavailable, running dashboard in read-only mode: {exc}")

if db_ready:
    conn = get_connection()

    try:
        portfolio = pd.read_sql("SELECT * FROM portfolio ORDER BY time", conn)
        transactions = pd.read_sql("SELECT * FROM transactions ORDER BY time", conn)
        positions = pd.read_sql(
            """
            SELECT *
            FROM position_snapshots
            WHERE time = (SELECT MAX(time) FROM position_snapshots)
            ORDER BY market_value DESC
            """,
            conn,
        )
        signals = pd.read_sql(
            """
            SELECT *
            FROM recommendation_signals
            ORDER BY ABS(score - CASE WHEN score >= 0 THEN 1 ELSE -1 END) ASC, ABS(score) DESC
            LIMIT 20
            """,
            conn,
        )
    except Exception:
        portfolio = pd.DataFrame()
        transactions = pd.DataFrame()
        positions = pd.DataFrame()
        signals = pd.DataFrame()
    finally:
        conn.close()
else:
    portfolio = pd.DataFrame()
    transactions = pd.DataFrame()
    positions = pd.DataFrame()
    signals = pd.DataFrame()

skeleton_placeholder.empty()

all_tickers = sorted(
    set(
        [str(t) for t in positions.get("ticker", pd.Series(dtype=str)).dropna().tolist()]
        + [str(t) for t in transactions.get("ticker", pd.Series(dtype=str)).dropna().tolist()]
        + [str(t) for t in signals.get("ticker", pd.Series(dtype=str)).dropna().tolist()]
    )
)

with st.sidebar:
    if logo_uri:
        st.markdown(
            f'<div class="kb-sidebar-brand"><img src="{logo_uri}" alt="kaibot"/><span>kaibot</span></div>',
            unsafe_allow_html=True,
        )
    st.subheader("At-a-Glance")
    auto_refresh = st.toggle("Auto-refresh", value=True, help="Poll database periodically for new worker updates.")
    refresh_seconds = st.slider("Refresh every (seconds)", min_value=5, max_value=300, value=10, step=5)
    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=int(refresh_seconds * 1000), key="dashboard_autorefresh")
        else:
            st.warning("`streamlit-autorefresh` is not installed; auto-refresh is disabled.")

    st.markdown("**Dashboard Filters**")
    selected_tickers = st.multiselect("Tickers", options=all_tickers, default=[])
    decision_filter = st.multiselect("Decisions", options=["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
    st.markdown("**Recommendation Trace**")
    trace_rows = load_jsonl_dict_rows("logs/recommendation_trace.jsonl")
    if trace_rows:
        explain_report = generate_explainability_report(trace_rows, max_examples=8)
        decision_counts = explain_report.get("decision_counts", {}) or {}
        st.caption(
            f"Entries: {int(explain_report.get('total_entries', 0))} | "
            f"BUY: {int(decision_counts.get('BUY', 0))} "
            f"SELL: {int(decision_counts.get('SELL', 0))} "
            f"HOLD: {int(decision_counts.get('HOLD', 0))}"
        )
        top_reasons = explain_report.get("top_reasons", [])[:6]
        if top_reasons:
            st.dataframe(
                pd.DataFrame(top_reasons, columns=["reason", "count"]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("No trace rows yet.")

    st.markdown("**Experiments**")
    exp = _load_latest_experiment_result()
    if exp:
        exp_payload, exp_path = exp
        best_variant = exp_payload.get("best_variant")
        st.caption(f"Latest: {exp_path.name}")
        if best_variant:
            st.caption(
                f"Best: {best_variant.get('variant', '-')} | "
                f"Test obj: {float(best_variant.get('test_objective', 0.0)):.4f}"
            )
        rows = exp_payload.get("rows", []) or []
        if rows:
            sidebar_exp = pd.DataFrame(rows).head(8)
            st.dataframe(sidebar_exp, use_container_width=True, hide_index=True)
    else:
        st.caption("No experiment artifacts yet.")

if selected_tickers:
    if not positions.empty and "ticker" in positions.columns:
        positions = positions[positions["ticker"].astype(str).isin(selected_tickers)]
    if not transactions.empty and "ticker" in transactions.columns:
        transactions = transactions[transactions["ticker"].astype(str).isin(selected_tickers)]
    if not signals.empty and "ticker" in signals.columns:
        signals = signals[signals["ticker"].astype(str).isin(selected_tickers)]

if decision_filter and not signals.empty and "decision" in signals.columns:
    signals = signals[signals["decision"].astype(str).str.upper().isin(decision_filter)]

# Risk Monitor (from trace)
trace_for_monitor = load_jsonl_dict_rows("logs/recommendation_trace.jsonl")
if trace_for_monitor:
    total_entries = max(len(trace_for_monitor), 1)
    guardrail_hits = 0
    veto_hits = 0
    high_vol_hits = 0
    low_conf_hold_hits = 0
    for row in trace_for_monitor:
        reasons = [str(r) for r in (row.get("decision_reasons", []) or [])]
        decision = str(row.get("decision", "HOLD")).upper()
        if any(r.startswith("guardrail:") for r in reasons):
            guardrail_hits += 1
        if any(r.startswith("veto:") for r in reasons):
            veto_hits += 1
        if "volatility:high" in reasons:
            high_vol_hits += 1
        if decision == "HOLD" and any(r.startswith("confidence:") for r in reasons):
            low_conf_hold_hits += 1

    st.subheader("Risk Monitor")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Guardrail Hit Rate", f"{100 * guardrail_hits / total_entries:.1f}%")
    r2.metric("Veto Rate", f"{100 * veto_hits / total_entries:.1f}%")
    r3.metric("High Volatility Rate", f"{100 * high_vol_hits / total_entries:.1f}%")
    r4.metric("Low-Conf HOLD Rate", f"{100 * low_conf_hold_hits / total_entries:.1f}%")

if not portfolio.empty:
    latest = float(portfolio["value"].iloc[-1])
    baseline_capital = float(TOP20_STARTING_CAPITAL)
    change_pct = ((latest - baseline_capital) / baseline_capital) * 100 if baseline_capital > 0 else 0.0
    running_peak = portfolio["value"].cummax()
    drawdown_series = (running_peak - portfolio["value"]) / running_peak.replace(0, pd.NA)
    latest_drawdown = float(drawdown_series.fillna(0).iloc[-1]) if not drawdown_series.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Portfolio Value", f"${latest:,.2f}")
    c2.metric("Net Return", f"{change_pct:+.2f}%")
    c3.metric("Tracked Cycles", f"{len(portfolio)}")
    c4.metric("Current Drawdown", f"{100 * latest_drawdown:.2f}%")

    g1, g2 = st.columns([2, 1])
    with g1:
        if not positions.empty and {"ticker", "market_value"}.issubset(set(positions.columns)):
            alloc_view_top = positions.sort_values("market_value", ascending=False).copy()
            pull = [0.06 if i < 2 else 0.0 for i in range(len(alloc_view_top))]
            alloc_top_fig = px.pie(
                alloc_view_top,
                names="ticker",
                values="market_value",
                hole=0.62,
                title="Current Allocation Mix",
                template=plot_template,
                color_discrete_sequence=pie_palette,
            )
            alloc_top_fig.update_traces(
                pull=pull,
                sort=False,
                textposition="inside",
                texttemplate="%{percent}",
                marker=dict(line=dict(color=card, width=2)),
                hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Weight: %{percent}<extra></extra>",
            )
            alloc_top_fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor=card,
                showlegend=True,
                legend=dict(orientation="v", y=0.95),
            )
            st.plotly_chart(alloc_top_fig, use_container_width=True)

        else:
            st.info("No allocation snapshot available yet.")
    with g2:
        if not positions.empty and {"ticker", "pnl_pct"}.issubset(set(positions.columns)):
            pnl_view_top = positions.copy()
            pnl_view_top["pnl_pct_display"] = pnl_view_top["pnl_pct"] * 100
            max_abs_pnl = float(pnl_view_top["pnl_pct_display"].abs().max()) if not pnl_view_top.empty else 0.0
            pnl_view_top["bar_color"] = pnl_view_top["pnl_pct_display"].apply(lambda v: _signed_bar_hex(v, max_abs_pnl))
            pnl_fig_top = px.bar(
                pnl_view_top.sort_values("pnl_pct_display", ascending=False),
                x="ticker",
                y="pnl_pct_display",
                color="ticker",
                title="Position P/L %",
                template=plot_template,
                color_discrete_map={row["ticker"]: row["bar_color"] for _, row in pnl_view_top.iterrows()},
            )
            pnl_fig_top.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>P/L: %{y:.2f}%<extra></extra>",
            )
            pnl_fig_top.update_layout(
                height=340,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor=card,
                plot_bgcolor=card,
                bargap=0.35,
                xaxis=dict(title="", showgrid=False, tickfont=dict(size=12)),
                yaxis=dict(
                    title="",
                    ticksuffix="%",
                    gridcolor=border,
                    zeroline=True,
                    zerolinecolor=text,
                    zerolinewidth=1.5,
                ),
            )
            st.plotly_chart(pnl_fig_top, use_container_width=True)
        else:
            st.info("No position P/L snapshot available yet.")
else:
    st.info("No portfolio data yet — worker has not run.")

st.subheader("Top Signals (Closest to Trade Trigger)")
if not signals.empty:
    signal_view = signals.copy()
    if "decision" in signal_view.columns:
        signal_view["decision"] = signal_view["decision"].astype(str).str.upper()
    signal_view["distance_to_trigger"] = signal_view["score"].apply(lambda v: abs(v - 1) if v >= 0 else abs(v + 1))
    signal_view = _with_logo_column(signal_view)
    signal_cols = [col for col in ["logo", "ticker", "decision", "score", "price", "distance_to_trigger", "time"] if col in signal_view.columns]
    signal_table = signal_view[signal_cols].sort_values(["distance_to_trigger", "score"], ascending=[True, False]).head(20)
    s1, s2 = st.columns([2, 1])
    with s1:
        signal_display = signal_table.copy()
        if "score" in signal_display.columns:
            signal_display["score"] = signal_display["score"].map(lambda v: f"{float(v):.3f}")
        if "price" in signal_display.columns:
            signal_display["price"] = signal_display["price"].map(lambda v: f"{float(v):.2f}")
        if "distance_to_trigger" in signal_display.columns:
            signal_display["distance_to_trigger"] = signal_display["distance_to_trigger"].map(lambda v: f"{float(v):.3f}")
        try:
            st.dataframe(
                signal_display,
                use_container_width=True,
                hide_index=True,
                column_config={"logo": st.column_config.ImageColumn("logo", width="small")},
            )
        except Exception:
            st.dataframe(signal_display, use_container_width=True, hide_index=True)
    with s2:
        signal_chart_df = signal_table.sort_values("score", ascending=False).head(12)
        sig_fig = px.bar(
            signal_chart_df,
            x="ticker",
            y="score",
            color="decision" if "decision" in signal_chart_df.columns else None,
            title="Signal Score Snapshot",
            template=plot_template,
        )
        sig_fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor=bg, plot_bgcolor=bg)
        st.plotly_chart(sig_fig, use_container_width=True)
else:
    st.caption("No latest signal data yet.")

with st.expander("Transaction History", expanded=False):
    if not transactions.empty:
        tx = _with_logo_column(transactions.copy())
        if "action" in tx.columns:
            tx["action"] = tx["action"].astype(str).str.upper()
        if "value" not in tx.columns and {"shares", "price"}.issubset(tx.columns):
            tx["value"] = tx["shares"] * tx["price"]

        tx_display = tx.copy()
        if "shares" in tx_display.columns:
            tx_display["shares"] = tx_display["shares"].map(lambda v: f"{float(v):.4f}")
        if "price" in tx_display.columns:
            tx_display["price"] = tx_display["price"].map(lambda v: f"{float(v):.2f}")
        if "value" in tx_display.columns:
            tx_display["value"] = tx_display["value"].map(lambda v: f"{float(v):.2f}")
        try:
            st.dataframe(
                tx_display,
                use_container_width=True,
                column_config={"logo": st.column_config.ImageColumn("logo", width="small")},
            )
        except Exception:
            st.dataframe(tx_display, use_container_width=True)
    else:
        st.write("No transactions yet.")
