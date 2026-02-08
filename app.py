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
    text_muted = "#9aa4b8"
    pos_neg_scale = ["#ffb3ba", "#ff8c96", "#a8e6b1", "#6bcf8b"]
    plot_template = "plotly_dark"
else:
    bg = "#f6f7fb"
    card = "#ffffff"
    border = "#e8ebf2"
    text_muted = "#5f6b7b"
    pos_neg_scale = ["#f6b0b0", "#ef7f8a", "#bfe8c7", "#82cfa0"]
    plot_template = "plotly_white"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: {bg};
      }}
      [data-testid="stMetric"] {{
        background: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 10px 14px;
      }}
      .block-container {{
        padding-top: 1.4rem;
      }}
      .bb-subtle {{
        color: {text_muted};
        margin-top: -0.25rem;
        margin-bottom: 1rem;
      }}
      h1, h2, h3 {{
        letter-spacing: -0.01em;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("assets/bloodbath_logo.svg", width=260)
st.title("Bloodbath")
st.markdown(
    '<p class="bb-subtle">Autonomous AI portfolio engine optimized for long-term capital growth.</p>',
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

    growth_fig = px.line(
        portfolio,
        x="time",
        y="value",
        title="Portfolio Growth Over Time",
        markers=True,
        template=plot_template,
    )
    growth_fig.update_layout(height=350)
    st.plotly_chart(growth_fig, use_container_width=True)
else:
    st.info("No portfolio data yet — worker has not run.")

st.subheader("Current Allocation & Position Performance")

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
            hole=0.48,
            title="Fund Allocation by Ticker",
            template=plot_template,
            color_discrete_sequence=[
                "#DDEAF7",
                "#E8E0F8",
                "#DDF2E3",
                "#FCE8D8",
                "#F9DDE3",
                "#E1ECFF",
            ],
        )
        alloc_fig.update_layout(height=360)
        st.plotly_chart(alloc_fig, use_container_width=True)

    with viz_col2:
        pnl_fig = px.bar(
            positions,
            x="ticker",
            y="pnl_pct_display",
            color="pnl_pct_display",
            title="Position P/L %",
            template=plot_template,
            color_continuous_scale=pos_neg_scale,
        )
        pnl_fig.update_layout(height=360, coloraxis_showscale=False)
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

with st.expander("Transaction History (expand to view)", expanded=False):
    if not transactions.empty:
        st.dataframe(transactions, use_container_width=True)
    else:
        st.write("No transactions yet.")
