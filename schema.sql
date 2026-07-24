-- =====================================================================
-- Event Ticket Dynamic Pricing — Database Schema (MySQL)
-- =====================================================================
-- Just 3 tables. events and customers each get linked into sales
-- using their ID — the same JOIN pattern you already know.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS ticket_shop;
USE ticket_shop;

DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS customers;

-- One row per concert / match / show
CREATE TABLE events (
    event_id     INT AUTO_INCREMENT PRIMARY KEY,
    event_name   VARCHAR(150) NOT NULL,
    venue        VARCHAR(150) NOT NULL,
    event_date   DATE NOT NULL,
    total_seats  INT NOT NULL,
    base_price   DECIMAL(10,2) NOT NULL
);

-- One row per person who ever bought a ticket
CREATE TABLE customers (
    customer_id  INT AUTO_INCREMENT PRIMARY KEY,
    name         VARCHAR(100) NOT NULL,
    city         VARCHAR(100)
);

-- One row per ticket purchase
CREATE TABLE sales (
    sale_id      INT AUTO_INCREMENT PRIMARY KEY,
    event_id     INT NOT NULL,
    customer_id  INT NOT NULL,
    sale_date    DATE NOT NULL,
    quantity     INT NOT NULL,
    price_paid   DECIMAL(10,2) NOT NULL,   -- price PER TICKET at the time of sale
    FOREIGN KEY (event_id) REFERENCES events(event_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
