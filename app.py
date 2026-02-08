import streamlit as st
import pandas as pd
import plotly.express as px

from src.db import get_connection, init_db

st.set_page_config(page_title="Bloodbath", page_icon="🩸", layout="wide")

theme_base = st.get_option("theme.base") or "light"
is_dark = theme_base.lower() == "dark"

if is_dark:
    bg = "#0f1115"
    card = "#171a21"
    border = "#232a35"
    text = "#e7edf7"
    panel = "#121722"
    plot_template = "plotly_dark"
else:
    bg = "#f6f7fb"
    card = "#ffffff"
    border = "#e8ebf2"
    text = "#1f2a44"
    panel = "#ffffff"
    plot_template = "plotly_white"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: {bg};
        color: {text};
      }}
      [data-testid="stHeader"] {{
        background: {bg};
        border-bottom: 1px solid {border};
      }}
      [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{
        visibility: hidden;
        height: 0;
        position: fixed;
      }}
      .block-container {{
        padding-top: 1rem;
        max-width: 1200px;
      }}
      [data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 10px 16px;
      }}
      .bb-topbar {{
        display: flex;
        align-items: center;
        gap: 12px;
        background: {panel};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 10px 16px;
        margin-bottom: 1rem;
      }}
      .bb-topbar-title {{
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.02em;
      }}
      h2, h3 {{
        letter-spacing: -0.01em;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

logo_col, title_col = st.columns([0.08, 0.92])
with logo_col:
    st.image("assets/bloodbath_logo.svg", width=58)
with title_col:
    st.markdown('<div class="bb-topbar"><span class="bb-topbar-title">Bloodbath</span></div>', unsafe_allow_html=True)

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
    growth_fig.update_traces(line_color="#f35f6d", fillcolor="rgba(243,95,109,0.18)")
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
                "#FAD2D6",
                "#FEE1C7",
                "#FDECB3",
                "#D8F0D2",
                "#D8E9FB",
                "#E7DDF9",
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
            color_continuous_scale=["#f5b6be", "#f7d8dc", "#d5edd9", "#9fd9ad"],
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
