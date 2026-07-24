-- =====================================================================
-- Analysis Queries — Event Ticket Dynamic Pricing
-- Run these directly in MySQL (or MySQL Workbench) after generate_data.py
-- =====================================================================
USE ticket_shop;

-- ---------------------------------------------------------------------
-- 1. Revenue and tickets sold, per event
-- ---------------------------------------------------------------------
SELECT
    e.event_name,
    e.venue,
    SUM(s.quantity)                    AS tickets_sold,
    SUM(s.quantity * s.price_paid)     AS total_revenue,
    ROUND(AVG(s.price_paid), 2)        AS avg_price_paid
FROM sales s
JOIN events e ON s.event_id = e.event_id
GROUP BY e.event_id
ORDER BY total_revenue DESC;

-- ---------------------------------------------------------------------
-- 2. Dynamic pricing revenue uplift vs. flat base price
--    (this is the headline number for the whole project)
-- ---------------------------------------------------------------------
SELECT
    e.event_name,
    SUM(s.quantity * s.price_paid)                     AS actual_revenue,
    SUM(s.quantity * e.base_price)                      AS baseline_revenue,
    ROUND(
        100.0 * (SUM(s.quantity * s.price_paid) - SUM(s.quantity * e.base_price))
        / SUM(s.quantity * e.base_price), 2
    )                                                    AS pct_uplift
FROM sales s
JOIN events e ON s.event_id = e.event_id
GROUP BY e.event_id
ORDER BY pct_uplift DESC;

-- ---------------------------------------------------------------------
-- 3. Daily sales trend (for the line chart on the dashboard)
-- ---------------------------------------------------------------------
SELECT
    sale_date,
    SUM(quantity)               AS tickets_sold,
    SUM(quantity * price_paid)  AS revenue
FROM sales
GROUP BY sale_date
ORDER BY sale_date;

-- ---------------------------------------------------------------------
-- 4. Top 10 customers by total spend
-- ---------------------------------------------------------------------
SELECT
    c.name,
    c.city,
    COUNT(s.sale_id)               AS num_purchases,
    SUM(s.quantity * s.price_paid) AS total_spent
FROM customers c
JOIN sales s ON c.customer_id = s.customer_id
GROUP BY c.customer_id
ORDER BY total_spent DESC
LIMIT 10;

-- ---------------------------------------------------------------------
-- 5. Sell-through rate per event (how full did it get?)
-- ---------------------------------------------------------------------
SELECT
    e.event_name,
    e.total_seats,
    COALESCE(SUM(s.quantity), 0)                              AS seats_sold,
    ROUND(100.0 * COALESCE(SUM(s.quantity), 0) / e.total_seats, 1) AS sell_through_pct
FROM events e
LEFT JOIN sales s ON e.event_id = s.event_id
GROUP BY e.event_id
ORDER BY sell_through_pct DESC;
