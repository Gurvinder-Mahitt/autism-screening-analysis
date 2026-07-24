-- =====================================================================
-- Event Ticket Dynamic Pricing — Database Schema (SQLite)
-- =====================================================================
-- SQLite version of schema.sql — same 3 tables, adapted for SQLite
-- syntax (AUTOINCREMENT, no DECIMAL type, etc.).
--
-- This file is for reference only. generate_data_sqlite.py creates
-- the schema automatically.
-- =====================================================================

DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS customers;

-- One row per concert / match / show
CREATE TABLE events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name   TEXT    NOT NULL,
    venue        TEXT    NOT NULL,
    event_date   TEXT    NOT NULL,
    total_seats  INTEGER NOT NULL,
    base_price   REAL    NOT NULL
);

-- One row per person who ever bought a ticket
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    city         TEXT
);

-- One row per ticket purchase
CREATE TABLE sales (
    sale_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     INTEGER NOT NULL,
    customer_id  INTEGER NOT NULL,
    sale_date    TEXT    NOT NULL,
    quantity     INTEGER NOT NULL,
    price_paid   REAL    NOT NULL,   -- price PER TICKET at the time of sale
    FOREIGN KEY (event_id) REFERENCES events(event_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
