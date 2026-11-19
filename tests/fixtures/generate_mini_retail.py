"""Generate tests/fixtures/mini_retail.csv: a small, fully SYNTHETIC table in the
same column layout as the UCI Online Retail file, with the same kinds of quirks.

It contains no UCI data. Run:  python tests/fixtures/generate_mini_retail.py
The output is deterministic (fixed seed).
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).with_name("mini_retail.csv")
COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("47566", "PARTY BUNTING", 4.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
    ("22960", "JAM MAKING SET WITH JARS", 4.25),
]
CUSTOMER_COUNTRY = {}


def main() -> None:
    rng = random.Random(20101201)
    countries = ["United Kingdom"] * 6 + ["France", "Germany", "EIRE"]
    customers = [str(12346 + i * 7) for i in range(30)]
    for c in customers:
        CUSTOMER_COUNTRY[c] = rng.choice(countries)
    start = datetime(2010, 12, 1, 8, 0, 0)
    rows: list[list[object]] = []

    def stamp() -> str:
        t = start + timedelta(minutes=rng.randint(0, 90 * 24 * 60))
        return t.strftime("%Y-%m-%d %H:%M:%S")

    invoice = 536365
    normal_invoices: list[tuple[str, str, str, str]] = []
    for _ in range(120):
        cust = rng.choice(customers)
        ts = stamp()
        country = CUSTOMER_COUNTRY[cust]
        cust_field = "" if rng.random() < 0.12 else f"{cust}.0"
        inv = str(invoice)
        invoice += 1
        normal_invoices.append((inv, ts, cust_field, country))
        for code, desc, price in rng.sample(PRODUCTS, rng.randint(1, 5)):
            qty = rng.choice([1, 2, 3, 4, 6, 8, 12, 24, 48])
            rows.append([inv, code, desc, qty, ts, price, cust_field, country])

    # Cancellations (C prefix, negative quantity), mostly of earlier invoices.
    for k in range(10):
        inv_ref = normal_invoices[k * 7]
        ts = datetime.strptime(inv_ref[1], "%Y-%m-%d %H:%M:%S") + timedelta(days=3 + k)
        code, desc, price = PRODUCTS[k % len(PRODUCTS)]
        cust = inv_ref[2]
        rows.append(
            [f"C{invoice}", code, desc, -rng.choice([1, 2, 6, 12]), ts, price, cust, inv_ref[3]]
        )
        invoice += 1

    # Hand-placed edge cases.
    ts = "2011-01-15 10:00:00"
    rows += [
        [f"{invoice}", "POST", "POSTAGE", 1, ts, 18.0, "12346.0", "United Kingdom"],
        [
            f"{invoice}",
            "85123A",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            5,
            ts,
            2.55,
            "12346.0",
            "United Kingdom",
        ],
        [f"C{invoice + 1}", "D", "Discount", -1, ts, 5.0, "12346.0", "United Kingdom"],
        [
            f"C{invoice + 1}",
            "22423",
            "REGENCY CAKESTAND 3 TIER",
            -1,
            ts,
            12.75,
            "12346.0",
            "United Kingdom",
        ],
        [f"{invoice + 2}", "22752", "damaged", -5, ts, 0.0, "", "United Kingdom"],
        [f"{invoice + 3}", "47566", "PARTY BUNTING", 3, ts, 0.0, "", "United Kingdom"],
        [f"A{invoice + 4}", "B", "Adjust bad debt", 1, ts, -500.0, "", "United Kingdom"],
        [f"{invoice + 5}", "m", "Manual", 1, ts, 25.0, "12353.0", "France"],
        [f"{invoice + 5}", "BANK CHARGES", "Bank Charges", 1, ts, 15.0, "12353.0", "France"],
        [
            f"{invoice + 6}",
            "20725",
            "  LUNCH BAG RED RETROSPOT  ",
            10,
            ts,
            1.65,
            "12360.0",
            "Unspecified",
        ],
        [f"{invoice + 7}", "84029G", "", 2, ts, 3.75, "12367.0", "Germany"],
        [f"C{invoice + 8}", "71053", "WHITE METAL LANTERN", 4, ts, 3.39, "12374.0", "EIRE"],
    ]
    # Exact duplicate lines (flagged, not removed).
    rows += [list(rows[0]), list(rows[0]), list(rows[5])]

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
