"""
dashboard.py
-------------
Interactive dashboard for the ticket shop. Reads directly from MySQL
and draws 4 charts + a KPI row — nothing more, nothing hidden.

Run:
    streamlit run dashboard.py
"""

import pandas as pd
import plotly.express as px
import pymysql
import streamlit as st

DB_CONFIG = dict(
    host="localhost",
    user="root",
    password="",          # <-- put your MySQL password here (same as generate_data.py)
    database="ticket_shop",
)

st.set_page_config(page_title="Ticket Shop — Dynamic Pricing", page_icon="🎟️", layout="wide")


def run_query(sql):
    conn = pymysql.connect(**DB_CONFIG)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
sales = run_query("""
    SELECT s.sale_date, s.quantity, s.price_paid, e.event_name, e.base_price
    FROM sales s JOIN events e ON s.event_id = e.event_id
""")
sales["revenue"] = sales["quantity"] * sales["price_paid"]
sales["baseline_revenue"] = sales["quantity"] * sales["base_price"]

# ---------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------
st.title("🎟️ Event Ticket Shop — Dynamic Pricing Dashboard")
st.caption("Tracks sales, revenue, and the impact of demand-based pricing.")

total_revenue = sales["revenue"].sum()
total_tickets = sales["quantity"].sum()
avg_price = sales["price_paid"].mean()
baseline_revenue = sales["baseline_revenue"].sum()
uplift_pct = 100 * (total_revenue - baseline_revenue) / baseline_revenue

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
k2.metric("Tickets Sold", f"{total_tickets:,.0f}")
k3.metric("Avg. Price Paid", f"₹{avg_price:,.0f}")
k4.metric("Dynamic Pricing Uplift", f"{uplift_pct:+.1f}%")

st.markdown("---")

# ---------------------------------------------------------------------
# Chart 1: Daily sales trend
# ---------------------------------------------------------------------
st.subheader("Daily Sales Trend")
daily = sales.groupby("sale_date").agg(revenue=("revenue", "sum"),
                                        tickets=("quantity", "sum")).reset_index()
fig1 = px.line(daily, x="sale_date", y="revenue", markers=True,
               labels={"sale_date": "Date", "revenue": "Revenue (₹)"})
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------
# Chart 2: Revenue by event
# ---------------------------------------------------------------------
st.subheader("Revenue by Event")
by_event = sales.groupby("event_name")["revenue"].sum().reset_index().sort_values("revenue")
fig2 = px.bar(by_event, x="revenue", y="event_name", orientation="h",
              labels={"revenue": "Revenue (₹)", "event_name": ""})
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------
# Chart 3: Revenue uplift per event (dynamic vs flat pricing)
# ---------------------------------------------------------------------
st.subheader("Dynamic Pricing Uplift by Event")
uplift = sales.groupby("event_name").agg(
    actual=("revenue", "sum"), baseline=("baseline_revenue", "sum")
).reset_index()
uplift["pct_uplift"] = 100 * (uplift["actual"] - uplift["baseline"]) / uplift["baseline"]
uplift = uplift.sort_values("pct_uplift")
fig3 = px.bar(uplift, x="pct_uplift", y="event_name", orientation="h",
              labels={"pct_uplift": "Revenue Uplift (%)", "event_name": ""},
              color="pct_uplift", color_continuous_scale="Purples")
fig3.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------
# Chart 4: Average price paid, by event (shows pricing varied at all)
# ---------------------------------------------------------------------
st.subheader("Average Price Paid vs. Base Price")
price_compare = sales.groupby("event_name").agg(
    avg_price_paid=("price_paid", "mean"), base_price=("base_price", "first")
).reset_index()
fig4 = px.bar(price_compare, x="event_name", y=["base_price", "avg_price_paid"],
              barmode="group", labels={"value": "Price (₹)", "event_name": ""})
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
st.caption("Built with Python, MySQL, Streamlit, and Plotly.")
