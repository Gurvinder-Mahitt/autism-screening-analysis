"""
generate_data_sqlite.py
------------------------
SQLite version of generate_data.py — same logic, zero setup.

Creates a local 'ticket_shop.db' file, builds the schema, and fills it
with the same realistic fake sales data using the dynamic pricing rule.

Run:
    python generate_data_sqlite.py
"""

import os
import random
import sqlite3
from datetime import date, timedelta

# ---------------------------------------------------------------------
# 1. Database setup — just a local file, no server needed
# ---------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticket_shop.db")

random.seed(42)
TODAY = date(2026, 7, 17)


def create_schema(cur):
    """Drop and recreate the 3 tables (events, customers, sales)."""
    cur.executescript("""
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE events (
            event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name   TEXT    NOT NULL,
            venue        TEXT    NOT NULL,
            event_date   TEXT    NOT NULL,
            total_seats  INTEGER NOT NULL,
            base_price   REAL    NOT NULL
        );

        CREATE TABLE customers (
            customer_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            city         TEXT
        );

        CREATE TABLE sales (
            sale_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id     INTEGER NOT NULL,
            customer_id  INTEGER NOT NULL,
            sale_date    TEXT    NOT NULL,
            quantity     INTEGER NOT NULL,
            price_paid   REAL    NOT NULL,
            FOREIGN KEY (event_id) REFERENCES events(event_id),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
    """)


# ---------------------------------------------------------------------
# 2. The dynamic pricing rule — the whole "brain" of the project
# ---------------------------------------------------------------------
def calculate_price(base_price, days_to_event, sale_window_days, sold_ratio):
    """
    price = base_price x (1 + urgency + scarcity)

    urgency : rises as the event gets closer (0 up to 0.5)
    scarcity: rises as more seats get sold   (0 up to 0.5)

    So price can go up to 2x base_price when the event is very close
    AND almost sold out, and stays near base_price when it's far away
    and mostly empty.
    """
    urgency = 0.5 * (1 - min(days_to_event / sale_window_days, 1))
    scarcity = 0.5 * min(sold_ratio, 1)
    multiplier = 1 + urgency + scarcity
    price = base_price * multiplier
    return round(price, 2)


# ---------------------------------------------------------------------
# 3. Sample data
# ---------------------------------------------------------------------
EVENTS = [
    # name, venue, days_from_today, total_seats, base_price
    ("Arijit Singh Live", "NSCI Dome, Mumbai", 40, 3000, 1000),
    ("Diljit Dosanjh Tour", "EKA Arena, Ahmedabad", 55, 2500, 900),
    ("Stand-up Comedy Night", "Guru Nanak Stadium, Ludhiana", 20, 800, 500),
    ("IPL Playoff Match", "Eden Gardens, Kolkata", 30, 5000, 1200),
    ("TechSummit 2026", "Sree Kanteerava, Bengaluru", 60, 1500, 700),
    ("AR Rahman Symphony", "JLN Stadium, Delhi", 25, 4000, 1100),
    ("Kabaddi League Finals", "NSCI Dome, Mumbai", 15, 2000, 600),
    ("Sunburn Music Festival", "EKA Arena, Ahmedabad", 70, 6000, 1300),
]

FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Simran", "Karan", "Neha", "Ishaan",
               "Ananya", "Gurpreet", "Kabir", "Diya", "Manpreet", "Arjun", "Sara"]
LAST_NAMES = ["Sharma", "Kaur", "Verma", "Singh", "Gupta", "Patel", "Reddy", "Nair"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Ludhiana", "Chandigarh", "Pune", "Kolkata"]

NUM_CUSTOMERS = 300


def main():
    # Remove old database file if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed old database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    create_schema(cur)
    print("Schema created.\n")

    # -----------------------------------------------------------------
    # Insert events
    # -----------------------------------------------------------------
    event_ids = []
    for name, venue, days_out, seats, base_price in EVENTS:
        event_date = TODAY + timedelta(days=days_out)
        cur.execute(
            "INSERT INTO events (event_name, venue, event_date, total_seats, base_price) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, venue, event_date.isoformat(), seats, base_price),
        )
        event_ids.append(cur.lastrowid)
    print(f"Inserted {len(event_ids)} events")

    # -----------------------------------------------------------------
    # Insert customers
    # -----------------------------------------------------------------
    customer_ids = []
    for _ in range(NUM_CUSTOMERS):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        city = random.choice(CITIES)
        cur.execute("INSERT INTO customers (name, city) VALUES (?, ?)", (name, city))
        customer_ids.append(cur.lastrowid)
    print(f"Inserted {len(customer_ids)} customers")

    # -----------------------------------------------------------------
    # Simulate daily sales for each event, using calculate_price()
    # -----------------------------------------------------------------
    total_sales = 0
    sales_batch = []
    SALE_WINDOW_DAYS = 90   # tickets always go on sale a fixed 90 days before the event

    for (name, venue, days_out, seats, base_price), event_id in zip(EVENTS, event_ids):
        event_date = TODAY + timedelta(days=days_out)
        sale_window_days = SALE_WINDOW_DAYS
        on_sale_date = event_date - timedelta(days=sale_window_days)
        # tickets can't have gone on sale before "today minus the window" — clip to today
        # if the on-sale date somehow lands in the future, there's simply no sales history yet
        if on_sale_date > TODAY:
            print(f"  (skipping {name} — on-sale date is still in the future)")
            continue

        sold_so_far = 0
        day = on_sale_date
        popularity = random.uniform(0.8, 1.2)          # some events are just more popular

        while day <= TODAY and sold_so_far < seats:
            days_to_event = (event_date - day).days
            sold_ratio = sold_so_far / seats

            price_today = calculate_price(base_price, days_to_event, sale_window_days, sold_ratio)

            # more people buy as the event gets closer (urgency drives real behaviour too)
            expected_sales_today = (seats / sale_window_days) * popularity
            if days_to_event <= 5:
                expected_sales_today *= 1.5
            tickets_today = min(int(random.gauss(expected_sales_today, expected_sales_today * 0.4)),
                                 seats - sold_so_far)
            tickets_today = max(tickets_today, 0)

            # turn today's tickets into a handful of separate purchases
            remaining = tickets_today
            while remaining > 0:
                qty = min(random.randint(1, 4), remaining)
                remaining -= qty
                customer_id = random.choice(customer_ids)
                sales_batch.append(
                    (event_id, customer_id, day.isoformat(), qty, price_today)
                )
                total_sales += 1

            sold_so_far += tickets_today
            day += timedelta(days=1)

    # Batch insert all sales for better performance
    cur.executemany(
        "INSERT INTO sales (event_id, customer_id, sale_date, quantity, price_paid) "
        "VALUES (?, ?, ?, ?, ?)",
        sales_batch,
    )

    conn.commit()
    print(f"Inserted {total_sales} sale records")
    print(f"\nDone! Data is ready in '{DB_PATH}'.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
