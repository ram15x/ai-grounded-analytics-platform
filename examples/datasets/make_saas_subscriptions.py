"""Writes a small SYNTHETIC subscription-billing table (invented, seeded), used to show that the
pipeline is not tied to the retail file. Usage: python make_saas_subscriptions.py out.csv"""

import csv
import random
import sys
from datetime import date, timedelta

PLANS = {"Starter": 19, "Team": 79, "Business": 249}
REGIONS = ["EMEA", "APAC", "AMER"]
random.seed(7)
rows = []
for i in range(1, 1201):
    start = date(2024, 1, 1) + timedelta(days=random.randint(0, 540))
    plan = random.choices(list(PLANS), weights=[6, 3, 1])[0]
    seats = random.randint(1, 5) if plan == "Starter" else random.randint(3, 40)
    churned = random.random() < 0.18
    end = start + timedelta(days=random.randint(30, 300)) if churned else None
    status = "churned" if churned else ("trial" if random.random() < 0.05 else "active")
    mrr = PLANS[plan] + max(0, seats - 3) * (PLANS[plan] // 10)
    rows.append(
        [
            f"SUB{i:05d}",
            start.isoformat(),
            end.isoformat() if end else "",
            plan,
            random.choice(REGIONS),
            status,
            seats,
            mrr,
            f"user{i}@example.com",
        ]
    )
with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "Subscription ID",
            "Start Date",
            "End Date",
            "Plan",
            "Region",
            "Status",
            "Seats",
            "MRR (USD)",
            "Billing Email",
        ]
    )
    w.writerows(rows)
