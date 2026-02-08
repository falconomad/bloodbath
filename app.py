import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from pathlib import Path

from src.db import get_connection, init_db

st.set_page_config(page_title="Bloodbath", page_icon="🩸", layout="wide")

logo_path = Path("assets/bloodbath_logo.svg")
logo_uri = ""
if logo_path.exists():
    logo_bytes = logo_path.read_bytes()
    logo_uri = f"data:image/svg+xml;base64,{base64.b64encode(logo_bytes).decode('utf-8')}"

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
    accent_soft = "rgba(77, 141, 255, 0.15)"
    metric_bg_1 = "#1b2333"
    metric_bg_2 = "#1b2a2a"
    metric_bg_3 = "#232033"
    plot_template = "plotly_dark"
else:
    bg = "#f3f4f6"
    card = "#ffffff"
    border = "#e3e6ec"
    text = "#141821"
    panel = "#ffffff"
    muted_text = "#6f7684"
    accent = "#2f6df6"
    accent_soft = "rgba(47, 109, 246, 0.08)"
    metric_bg_1 = "#f7f9ff"
    metric_bg_2 = "#f5faf8"
    metric_bg_3 = "#f8f7ff"
    plot_template = "plotly_white"

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
      [data-testid="stHeader"]::before {{
        content: "";
        position: absolute;
        left: max(calc((100vw - 1200px) / 2 + 1rem), 1rem);
        top: 54%;
        width: 10rem;
        height: 2.7rem;
        transform: translateY(-50%);
        border-radius: 0.6rem;
        background-image: url('{logo_uri}');
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
      }}
      [data-testid="stHeader"]::after {{
        content: "Bloodbath";
        position: absolute;
        left: max(calc((100vw - 1200px) / 2 + 4.1rem), 4.1rem);
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.62rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: {text};
        line-height: 1;
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
        animation: bb-fade-rise 0.55s ease both;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
      }}
      [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(18, 24, 33, 0.09);
      }}
      /* Portfolio metric cards (first row, 3 columns) */
      [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] {{
        background: {metric_bg_1};
        animation: bb-fade-rise 0.55s ease both, bb-bg-fade 4.4s ease-in-out 0.2s infinite alternate;
      }}
      [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {{
        background: {metric_bg_2};
        animation: bb-fade-rise 0.55s ease both, bb-bg-fade 4.4s ease-in-out 0.45s infinite alternate;
      }}
      [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {{
        background: {metric_bg_3};
        animation: bb-fade-rise 0.55s ease both, bb-bg-fade 4.4s ease-in-out 0.7s infinite alternate;
      }}
      [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stExpander"] {{
        animation: bb-fade-rise 0.7s ease both;
        background: {card};
        border-radius: 16px;
      }}
      [data-testid="stPlotlyChart"] > div,
      [data-testid="stDataFrame"] > div,
      [data-testid="stExpander"] {{
        border: 1px solid {border};
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(18, 24, 33, 0.04);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
      }}
      [data-testid="stPlotlyChart"]:hover > div,
      [data-testid="stDataFrame"]:hover > div,
      [data-testid="stExpander"]:hover {{
        box-shadow: 0 14px 28px rgba(18, 24, 33, 0.08);
      }}
      .stDataFrame [data-testid="stDataFrameResizable"] {{
        border-radius: 16px;
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
        animation-delay: 0.08s;
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
    </style>
    """,
    unsafe_allow_html=True,
)

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
    except Exception:
        portfolio = pd.DataFrame()
        transactions = pd.DataFrame()
        positions = pd.DataFrame()
    finally:
        conn.close()
else:
    portfolio = pd.DataFrame()
    transactions = pd.DataFrame()
    positions = pd.DataFrame()

if not portfolio.empty:
    latest = float(portfolio["value"].iloc[-1])
    start = float(portfolio["value"].iloc[0])
    change_pct = ((latest - start) / start) * 100 if start > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Portfolio Value", f"${latest:,.2f}")
    c2.metric("Net Return", f"{change_pct:+.2f}%")
    c3.metric("Tracked Cycles", f"{len(portfolio)}")

    growth_fig = px.area(
        portfolio,
        x="time",
        y="value",
        title="Portfolio Growth",
        template=plot_template,
    )
    growth_fig.update_traces(line_color=accent, fillcolor=accent_soft)
    growth_fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor=card, plot_bgcolor=card)
    st.plotly_chart(growth_fig, use_container_width=True)
else:
    st.info("No portfolio data yet — worker has not run.")

st.subheader("Current Allocation")

if not positions.empty:
    positions = positions.copy()
    positions["allocation_pct"] = positions["allocation"] * 100
    positions["pnl_pct_display"] = positions["pnl_pct"] * 100

    viz_col1, viz_col2 = st.columns([1, 1])

    with viz_col1:
        alloc_fig = px.pie(
            positions,
            names="ticker",
            values="market_value",
            hole=0.52,
            title="Allocation Mix",
            template=plot_template,
            color_discrete_sequence=[
                "#c8dbff",
                "#d4e3ff",
                "#dae8ff",
                "#e3edff",
                "#eaf2ff",
                "#f2f6ff",
            ],
        )
        alloc_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), paper_bgcolor=card)
        st.plotly_chart(alloc_fig, use_container_width=True)

    with viz_col2:
        pnl_fig = px.bar(
            positions,
            x="ticker",
            y="pnl_pct_display",
            color="pnl_pct_display",
            title="Position P/L %",
            template=plot_template,
            color_continuous_scale=["#d4def7", "#c8d8fb", "#aac5fa", accent],
        )
        pnl_fig.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=20), paper_bgcolor=card, plot_bgcolor=card)
        st.plotly_chart(pnl_fig, use_container_width=True)

    clean_table = positions[
        [
            "ticker",
            "shares",
            "avg_cost",
            "current_price",
            "market_value",
            "allocation_pct",
            "pnl",
            "pnl_pct_display",
        ]
    ].rename(
        columns={
            "avg_cost": "avg_cost($)",
            "current_price": "price($)",
            "market_value": "market_value($)",
            "allocation_pct": "allocation(%)",
            "pnl": "pnl($)",
            "pnl_pct_display": "pnl(%)",
        }
    )

    st.dataframe(
        clean_table.style.format(
            {
                "avg_cost($)": "{:.2f}",
                "price($)": "{:.2f}",
                "market_value($)": "{:.2f}",
                "allocation(%)": "{:.2f}",
                "pnl($)": "{:.2f}",
                "pnl(%)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
else:
    st.info("No open positions snapshot available yet.")

with st.expander("Transaction History", expanded=False):
    if not transactions.empty:
        st.dataframe(transactions, use_container_width=True)
    else:
        st.write("No transactions yet.")
