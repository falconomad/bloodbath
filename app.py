
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.title("Autonomous AI Fund Manager – Top 20 S&P 500")

conn = sqlite3.connect("db/portfolio.db")

portfolio = pd.read_sql("SELECT * FROM portfolio", conn)
transactions = pd.read_sql("SELECT * FROM transactions", conn)

if not portfolio.empty:
    latest = portfolio["value"].iloc[-1]
    start = portfolio["value"].iloc[0]
    change = ((latest - start) / start) * 100

    st.metric("Current Portfolio Value", f"{latest:.2f}")
    st.metric("Total Change %", f"{change:.2f}%")

    fig = px.line(portfolio, x="time", y="value", title="Portfolio Growth")
    st.plotly_chart(fig)
else:
    st.write("No data yet – start worker first.")

st.subheader("Transaction History")

if not transactions.empty:
    st.dataframe(transactions)
else:
    st.write("No transactions yet.")
