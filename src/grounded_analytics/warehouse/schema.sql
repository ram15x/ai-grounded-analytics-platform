-- Warehouse schema for the UCI Online Retail dataset.
-- `grounded load` drops and recreates these two schemas. Nothing outside them is touched.
DROP SCHEMA IF EXISTS analytics CASCADE;
DROP SCHEMA IF EXISTS warehouse CASCADE;
CREATE SCHEMA warehouse;
CREATE SCHEMA analytics;

CREATE TABLE warehouse.dim_customer (
    customer_key integer PRIMARY KEY,
    customer_id  text NOT NULL UNIQUE,
    country      text NOT NULL
);

CREATE TABLE warehouse.dim_product (
    product_key  integer PRIMARY KEY,
    stock_code   text NOT NULL UNIQUE,
    description  text,
    product_kind text NOT NULL CHECK (product_kind IN ('merchandise', 'non_merchandise'))
);

CREATE TABLE warehouse.fact_invoice (
    invoice_key     integer PRIMARY KEY,
    invoice_no      text NOT NULL UNIQUE,
    invoice_ts      timestamp NOT NULL,
    customer_key    integer REFERENCES warehouse.dim_customer (customer_key),
    country         text NOT NULL,
    is_cancellation boolean NOT NULL,
    n_lines         integer NOT NULL
);

CREATE TABLE warehouse.fact_invoice_lines (
    line_key           integer PRIMARY KEY,
    invoice_key        integer NOT NULL REFERENCES warehouse.fact_invoice (invoice_key),
    product_key        integer NOT NULL REFERENCES warehouse.dim_product (product_key),
    line_no            integer NOT NULL,
    quantity           integer NOT NULL,
    unit_price         numeric(14, 4) NOT NULL,
    line_amount        numeric(18, 4) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    line_type          text NOT NULL
        CHECK (line_type IN ('sale', 'cancellation', 'discount', 'adjustment')),
    is_exact_duplicate boolean NOT NULL,
    UNIQUE (invoice_key, line_no)
);

CREATE INDEX ix_lines_invoice ON warehouse.fact_invoice_lines (invoice_key);
CREATE INDEX ix_lines_product ON warehouse.fact_invoice_lines (product_key);
CREATE INDEX ix_invoice_customer ON warehouse.fact_invoice (customer_key);
CREATE INDEX ix_invoice_ts ON warehouse.fact_invoice (invoice_ts);

-- Exposed, read-only surface. The read-only role sees only this schema.
CREATE VIEW analytics.sales_lines AS
SELECT
    l.line_key,
    i.invoice_no,
    i.invoice_ts,
    i.invoice_ts::date AS invoice_date,
    date_trunc('month', i.invoice_ts)::date AS invoice_month,
    i.country,
    c.customer_id,
    p.stock_code,
    p.description AS product_description,
    p.product_kind,
    l.quantity,
    l.unit_price,
    l.line_amount,
    l.line_type,
    l.is_exact_duplicate
FROM warehouse.fact_invoice_lines AS l
JOIN warehouse.fact_invoice AS i ON i.invoice_key = l.invoice_key
JOIN warehouse.dim_product AS p ON p.product_key = l.product_key
LEFT JOIN warehouse.dim_customer AS c ON c.customer_key = i.customer_key;

CREATE VIEW analytics.products AS
SELECT stock_code, description, product_kind FROM warehouse.dim_product;

CREATE VIEW analytics.customers AS
SELECT customer_id, country FROM warehouse.dim_customer;

CREATE VIEW analytics.invoices AS
SELECT i.invoice_no, i.invoice_ts, c.customer_id, i.country, i.is_cancellation, i.n_lines
FROM warehouse.fact_invoice AS i
LEFT JOIN warehouse.dim_customer AS c ON c.customer_key = i.customer_key;
