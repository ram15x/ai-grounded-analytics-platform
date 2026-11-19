# Data profile and data-quality decisions

Source: UCI Online Retail (Chen, D., 2015, DOI 10.24432/C5BW33, CC BY 4.0). UK online gift
retailer, invoice lines from 2010-12-01 to 2011-12-09. Raw columns: InvoiceNo, StockCode,
Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

## Status of the numbers in this file

**The real dataset has NOT been loaded yet.** The build environment could not reach the UCI
host (HTTP 403 from its network proxy). Everything was built and tested against a 368-row
**synthetic** fixture (`tests/fixtures/mini_retail.csv`) that mimics the known quirks. So:

- The *decisions* below are implemented and unit-tested on the fixture.
- The *measured counts on the real file* are **pending**. Run `va load --download` (or
  `va load --source <file>`) then `va profile <file>` and paste the output into the table
  marked PENDING. Figures from the public dataset that I have not measured myself are not quoted here.
- The fixture numbers shown are labelled SYNTHETIC and say nothing about the real data.

## Warehouse tables

| Table | Grain | Notes |
|---|---|---|
| `warehouse.dim_customer` | one row per CustomerID | cleaned id (`12346.0` becomes `12346`); country = most frequent country for that customer |
| `warehouse.dim_product` | one row per stock code | upper-cased code; modal description; `product_kind` |
| `warehouse.fact_invoice` | one row per invoice | timestamp, country, customer, cancelled flag |
| `warehouse.fact_invoice_lines` | one row per source row | `line_key` = 1-based source row; `line_amount` = quantity x unit_price (generated column) |

Queries go through read-only views in schema `analytics` (`sales_lines`, `products`, `customers`,
`invoices`), never the base tables.

## Data-quality decisions

| # | Issue | Decision | Why |
|---|---|---|---|
| 1 | Invoice numbers starting `C` (cancellations) | Kept. Negative-quantity lines on a C invoice become `line_type = cancellation` | They are real returns; net revenue must subtract them and gross must not |
| 2 | Negative quantity without a C prefix | `adjustment`, excluded from every metric | Not a customer cancellation; usually stock write-offs |
| 3 | C invoice with positive quantity | `adjustment` | Contradictory row, not trusted |
| 4 | Zero or negative unit price | `adjustment` (excluded) | Not sales; typically manual adjustments or bad-debt entries |
| 5 | Stock code `D` (discount) with positive price | `discount` (own type, counted in net revenue, not gross) | Keeps discounts visible rather than silently mixed into sales |
| 6 | Missing CustomerID | Kept; `customer_key` NULL | Dropping loses revenue. Customer metrics (`active_customers`, `revenue_per_customer`) ignore these rows; `identified_gross_revenue` exists so the gap is explicit |
| 7 | Exact duplicate rows | Flagged (`is_exact_duplicate`), **not** removed | A repeated line can be a genuine repeat scan; removal would be a silent guess. Metrics currently include them |
| 8 | Non-merchandise codes (POST, M, DOT, BANK CHARGES, ...) | `product_kind = non_merchandise` (regex `^[0-9]{5}[A-Z]{0,3}$` = merchandise) | Lets questions exclude postage and fees without hiding them |
| 9 | Mixed-case stock codes | Upper-cased; collisions counted in the profile | Avoids two products for one code |
| 10 | Several descriptions per code | Modal description wins; count reported | Descriptions drift over time |
| 11 | Missing country | `Unknown` | Keeps rows groupable |
| 12 | Non-standard invoice numbers | Kept and counted | Never dropped for format alone |
| 13 | Customers in more than one country | Reported; invoice country is used for country metrics | The invoice, not the customer, carries the sale |
| 14 | Whitespace in text fields | Trimmed | |

The line-type rules are applied in this order: stock code D with price > 0 is a discount, else an
adjustment; price <= 0 is an adjustment; C invoice with qty < 0 is a cancellation, else an
adjustment; qty <= 0 is an adjustment; otherwise a sale.

Reconciliation on every load: table row counts equal the source, and the sum of `line_amount`
equals the sum computed from the raw file (see `reconcile` in `warehouse/load.py`).

## SYNTHETIC fixture (for illustration only, not the real data)

| Measure | Fixture value |
|---|---|
| Raw rows | 368 |
| Invoices / customers / products | 139 / 28 / 15 |
| Exact duplicate rows | 3 |
| Rows without CustomerID | 40 (19 invoices) |
| Line types | sale 352, cancellation 11, discount 1, adjustment 4 |
| Rows with zero / negative price | 2 / 1 |
| Date range | 2010-12-01 to 2011-02-28 |

## PENDING: measured on the real file

| Measure | Value |
|---|---|
| Raw rows | PENDING |
| Invoices / customers / products | PENDING |
| Exact duplicate rows | PENDING |
| Rows without CustomerID | PENDING |
| Rows on C invoices | PENDING |
| Line types (sale / cancellation / discount / adjustment) | PENDING |
| Rows with zero / negative price | PENDING |
| Net revenue / gross revenue / cancelled revenue | PENDING |

Producing this table: `va profile <file>` for the counts; `va load` for the database totals.
After running it, review decisions 2-7 against the real numbers; they may need revisiting.
