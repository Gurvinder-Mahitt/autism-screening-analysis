# 🎟️ Event Ticket Dynamic Pricing

A SQL + Python project that simulates an event ticket shop where prices go up automatically as an event gets closer and seats start selling out — instead of charging one flat price the whole way through.

**Headline result:** across 8 simulated events, dynamic pricing brought in **9.9% to 39.7% more revenue** than if every ticket had just been sold at a flat base price.

---

## Why I built this

Most beginner SQL/Python projects use a static dataset and stop at "here's a chart." I wanted to build something with actual logic behind it — a pricing rule that reacts to real signals (how close the event is, how many seats are left) — and then prove, with numbers, that the rule actually does something useful.

So this isn't just a database of ticket sales. The sales themselves were generated *using* the pricing rule, day by day, the same way a real system would set prices as tickets sold.

## 🧠 The pricing rule, in one line

```
price = base_price × (1 + urgency + scarcity)
```

- **Urgency** climbs from 0 to 0.5 as the event date approaches
- **Scarcity** climbs from 0 to 0.5 as more seats sell
- So price ranges from 1x base price (far out, empty) up to 2x base price (last few days, nearly sold out)

**Worked example:** ₹500 base price, 10 days left in a 90-day sale window, 60% sold:
```
urgency  = 0.5 × (1 - 10/90) = 0.444
scarcity = 0.5 × 0.60        = 0.300
price    = 500 × (1 + 0.444 + 0.300) = ₹872
```

## 📊 What the simulation actually produced

Ran this against 8 events (concerts, a comedy night, a cricket match, a music festival) and ~5,300 individual ticket sales. Here's how much more each event earned with dynamic pricing vs. a flat price the whole way:

| Event | Revenue Uplift |
|---|---|
| Kabaddi League Finals | +39.7% |
| AR Rahman Symphony | +37.7% |
| Stand-up Comedy Night | +34.2% |
| IPL Playoff Match | +32.3% |
| Arijit Singh Live | +29.4% |
| Diljit Dosanjh Tour | +19.9% |
| TechSummit 2026 | +16.2% |
| Sunburn Music Festival | +9.9% |

Makes sense that shorter sale-windows / high-urgency events (Kabaddi Finals had just 15 days to sell out) show the biggest uplift — less time for prices to sit near the base rate.

## 🖼️ Screenshots

**KPI row** — the headline numbers at a glance: ₹1.72 crore total revenue, 13,054 tickets sold, and an overall **+30% uplift** from dynamic pricing:

![KPI row](screenshots/kpi-row.png)

**Daily sales trend** — revenue climbs as more events enter their final, high-urgency sale days:

![Daily sales trend](screenshots/daily-sales-trend.png)

**Revenue by event** — IPL Playoff Match and AR Rahman Symphony brought in the most, driven by seat count and base price:

![Revenue by event](screenshots/revenue-by-event.png)

**Dynamic pricing uplift by event** — the headline chart. Kabaddi League Finals (shortest sale window, 15 days) saw the biggest lift at +39.7%:

![Uplift by event](screenshots/uplift-by-event.png)

**Average price paid vs. base price** — confirms the pricing rule is actually doing something, not just sitting on paper. Every event's average paid price sits above its base price:

![Price paid vs base price](screenshots/price-paid-vs-base.png)

## 🗄️ The data model

Three tables, joined the standard way:

```
events(event_id, event_name, venue, event_date, total_seats, base_price)
customers(customer_id, name, city)
sales(sale_id, event_id, customer_id, sale_date, quantity, price_paid)
```

## 📁 What's in this repo

```
├── ticket_shop.db            → pre-built SQLite database (just run the dashboard, no setup)
├── schema_sqlite.sql          → table definitions (SQLite version, for reference)
├── generate_data_sqlite.py    → simulates sales + writes ticket_shop.db (SQLite, zero setup)
├── dashboard_sqlite.py        → Streamlit dashboard, reads from ticket_shop.db
│
├── schema.sql                 → table definitions (MySQL version)
├── generate_data.py           → same simulation, for a real MySQL server
├── dashboard.py                → same dashboard, reads from MySQL
├── analysis_queries.sql       → 5 SQL questions answered directly against the data
│
├── requirements.txt
├── screenshots/
└── README.md
```

Two parallel versions on purpose: the **SQLite version** runs immediately with zero setup (good for anyone just cloning the repo to look), and the **MySQL version** is what I'd actually use with a real server, and shows I can work with both.

## 🚀 Quick start (easiest — SQLite, no install needed)

```bash
pip install -r requirements.txt
streamlit run dashboard_sqlite.py
```
Opens at `http://localhost:8501`. The database is already built and included (`ticket_shop.db`), so this just works out of the box.

Want to regenerate the data yourself (new random sales)?
```bash
python generate_data_sqlite.py
streamlit run dashboard_sqlite.py
```

## 🚀 Full setup (MySQL version)

```bash
pip install -r requirements.txt
mysql -u root -p < schema.sql
```
Then open `generate_data.py` and `dashboard.py` and set your MySQL password in `DB_CONFIG` near the top of each file.

```bash
python generate_data.py          # simulates ~5,000 sales across 8 events
mysql -u root -p ticket_shop < analysis_queries.sql   # optional: pure SQL analysis
streamlit run dashboard.py
```

## 📈 The dashboard

4 charts + a KPI row:
- **KPI row** — total revenue, tickets sold, average price paid, overall pricing uplift %
- **Daily sales trend** — revenue over time
- **Revenue by event** — which events made the most money
- **Dynamic pricing uplift by event** — the headline chart, per event
- **Average price paid vs. base price** — shows the pricing rule actually moved prices, not just theory

## 🔍 The 5 analysis queries (`analysis_queries.sql`)

1. Revenue and tickets sold per event
2. **Dynamic pricing revenue uplift vs. flat base price** — the headline metric
3. Daily sales trend
4. Top 10 customers by spend
5. Sell-through rate per event (how close to sold out did each event get)

## 🛠️ Built with

SQL (MySQL + SQLite) · Python · Streamlit · Plotly · PyMySQL

## 💡 What I'd add next

- A `price_history` table logging every price change over time, not just the price at the moment someone bought
- A third pricing factor — day-of-week demand patterns
- Deploying the dashboard live on Streamlit Community Cloud

## 👤 About

Built by Gurvinder — mining engineering background, building out SQL, Python, and data analytics skills for a move into analytics/consulting.
