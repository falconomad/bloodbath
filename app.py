import streamlit as st
import pandas as pd
import plotly.express as px

from src.db import get_connection, init_db

# Initialize tables if not already present
init_db()

st.title("Autonomous AI Fund Manager – Top 20 S&P 500")

# Connect to PostgreSQL
conn = get_connection()

# Read data from database
portfolio = pd.read_sql("SELECT * FROM portfolio ORDER BY time", conn)
transactions = pd.read_sql("SELECT * FROM transactions ORDER BY time", conn)

if not portfolio.empty:
    latest = portfolio["value"].iloc[-1]
    start = portfolio["value"].iloc[0]
    change = ((latest - start) / start) * 100

    st.metric("Current Portfolio Value", f"{latest:.2f}")
    st.metric("Total Change %", f"{change:.2f}%")

    fig = px.line(
        portfolio,
        x="time",
        y="value",
        title="Portfolio Growth Over Time"
    )
    st.plotly_chart(fig)

else:
    st.write("No data yet – worker has not run.")

st.subheader("Transaction History")

if not transactions.empty:
    st.dataframe(transactions)
else:
    st.write("No transactions yet.")

conn.close()
