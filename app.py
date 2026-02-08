import streamlit as st
import pandas as pd
import plotly.express as px

from src.db import get_connection, init_db

st.set_page_config(page_title="Bloodbath", page_icon="🩸", layout="wide")

st.image("assets/bloodbath_logo.svg", width=420)
st.title("Bloodbath")
st.caption("Autonomous AI portfolio engine optimized for long-term capital growth.")

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
            hole=0.45,
            title="Fund Allocation by Ticker",
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
            color_continuous_scale=["#a30000", "#ff6666", "#6fdc8c", "#188038"],
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
