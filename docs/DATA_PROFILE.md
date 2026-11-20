# Data profile and data-quality decisions

Source: UCI Online Retail (Chen, D., 2015, DOI 10.24432/C5BW33, CC BY 4.0). UK online gift
retailer, invoice lines from 2010-12-01 to 2011-12-09. Raw columns: InvoiceNo, StockCode,
Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

## Status of the numbers in this file

Measured on the real file on 2026-11-20: `va load`, `va profile` and `va profile-db`, run twice
independently (build chat and CTO chat) with byte-identical output. File: `Online Retail.xlsx`,
sha256 `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`, 541,909 rows. The file
itself is not in the repository (git-ignored). Tests still run on a 368-row synthetic fixture
(`tests/fixtures/mini_retail.csv`); nothing below comes from the fixture.

## Warehouse tables

| Table | Grain | Notes |
|---|---|---|
| `warehouse.dim_customer` | one row per CustomerID | cleaned id (`12346.0` becomes `12346`); country = most frequent country for that customer |
| `warehouse.dim_product` | one row per stock code | upper-cased code; modal description; `product_kind` |
| `warehouse.fact_invoice` | one row per invoice | timestamp, country, customer, cancelled flag |
| `warehouse.fact_invoice_lines` | one row per source row | `line_key` = 1-based source row; `line_amount` = quantity x unit_price (generated column) |

Queries go through read-only views in schema `analytics` (`sales_lines`, `products`, `customers`,
`invoices`), never the base tables.

## Data-quality decisions, with measured size

Percentages are of the 541,909 raw rows unless stated.

| # | Issue | Decision | Measured on the real file |
|---|---|---|---|
| 1 | Invoice numbers starting `C` (cancellations) | Kept. Negative-quantity lines on a C invoice become `cancellation` | 9,288 rows on C invoices: 9,211 cancellation (1.70%) and 77 discount. No C invoice has a positive quantity |
| 2 | Negative quantity without a C prefix | `adjustment`, excluded from every metric | 1,336 rows. All of them also have price <= 0, so rule 4 already classes them |
| 3 | C invoice with positive quantity | `adjustment` | 0 rows. The rule never fires on this file |
| 4 | Zero or negative unit price | `adjustment` (excluded) | 2,515 zero-price and 2 negative-price rows. 2,517 adjustments in total (0.46%), net amount -22,124.12 (none counted in any metric) |
| 5 | Stock code `D` (discount) | `discount` (own type, in net revenue, not in gross) | 77 rows, 5,696.22 |
| 6 | Missing CustomerID | Kept; `customer_key` NULL | 135,080 rows (24.93%) on 3,710 of 25,900 invoices (14.32%). Net revenue from identified customers is 8,300,065.81 (gold T1-15), so 1,469,806.24 (15.04%) has no customer |
| 7 | Exact duplicate rows | Flagged, **not** removed (owner decision, see below) | 5,268 rows (0.97%): 5,226 sale, 37 cancellation, 5 adjustment |
| 8 | Non-merchandise codes (POST, M, DOT, BANK CHARGES, ...) | `product_kind = non_merchandise` (regex `^[0-9]{5}[A-Z]{0,3}$` = merchandise) | 2,995 rows (0.55%); merchandise 538,914. Gross revenue from merchandise only is 10,271,034.61 of 10,666,684.54 |
| 9 | Mixed-case stock codes | Upper-cased; collisions counted | 1,977 rows changed; 112 code collisions |
| 10 | Several descriptions per code | Modal description wins; count reported | 642 products with more than one description; 110 with none |
| 11 | Missing country | `Unknown` | 0 rows. The rule never fires on this file |
| 12 | Non-standard invoice numbers | Kept and counted | 3 rows (A563185, A563186, A563187) |
| 13 | Customers in more than one country | Reported; invoice country used | 8 customers; 0 invoices with more than one country or customer |
| 14 | Whitespace in text fields | Trimmed | 113,452 descriptions changed (20.94%) |
| 15 | Invoices with more than one timestamp | Reported only | 43 invoices |
| 16 | Missing description | Kept | 1,454 rows |

Review of decisions 2-7 against the real numbers: no decision was changed. Rules 3 and 11 never
fire on this file; rule 2 is subsumed by rule 4. They stay as guards for other loads.

The line-type rules are applied in this order: stock code D with price > 0 is a discount, else an
adjustment; price <= 0 is an adjustment; C invoice with qty < 0 is a cancellation, else an
adjustment; qty <= 0 is an adjustment; otherwise a sale.

Reconciliation on every load: table row counts equal the source, and the sum of `line_amount`
equals the sum computed from the raw file (see `reconcile` in `warehouse/load.py`).

## Measured on the real file

| Measure | Value |
|---|---|
| Raw rows | 541,909 |
| Invoices / customers / products | 25,900 / 4,372 / 3,958 |
| Date range | 2010-12-01 08:26:00 to 2011-12-09 12:50:00 |
| Exact duplicate rows | 5,268 |
| Rows without CustomerID | 135,080 |
| Rows on C invoices | 9,288 |
| Line types (sale / cancellation / discount / adjustment) | 530,104 / 9,211 / 77 / 2,517 |
| Rows with zero / negative price | 2,515 / 2 |
| Gross revenue / cancelled revenue / discounts | 10,666,684.54 / 891,116.27 / 5,696.22 |
| Net revenue | 9,769,872.05 |
| Units sold (merchandise sale lines) | 5,576,996 |
| Orders / active customers | 19,960 / 4,338 |
| Average order value / revenue per customer | 534.40 / 2,054.27 |
| Sum of all line amounts (reconciles to the raw file) | 9,747,747.93 (= net revenue less 22,124.12 of adjustments) |

### Sensitivity: exact duplicates (owner decision: keep them counted)

| Metric | With duplicates (as loaded) | Without | Change |
|---|---|---|---|
| Net revenue | 9,769,872.05 | 9,748,131.07 | -21,740.98 (-0.22%) |
| Gross revenue | 10,666,684.54 | 10,642,110.80 | -0.23% |
| Cancelled revenue | 891,116.27 | 888,283.51 | -0.32% |
| Units sold | 5,576,996 | 5,561,280 | -15,716 (-0.28%) |
| Orders / active customers | 19,960 / 4,338 | 19,960 / 4,338 | none |

A repeated line can be a genuine repeat scan, and removal would be a silent guess, so they stay.
`va profile-db` reproduces this table at any time.

Reproduce: `va load --source "Online Retail.xlsx"`, `va profile "Online Retail.xlsx"`, `va profile-db`.
