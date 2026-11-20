# How revenue is defined in reports

SYNTHETIC DOCUMENT written for the Verified Analyst test corpus.

Net revenue is gross sales less cancellations and discounts. Gross revenue counts every sale line at
quantity multiplied by unit price, before cancellations. Invoices whose number starts with the letter
C are cancellations; their lines carry negative quantities.

Postage and manual adjustment codes are not merchandise. Reports of units sold count merchandise
lines only. An order is counted once per invoice number, so an order with many lines is one order.
Exact duplicate rows in the raw extract are reported separately and are not silently removed.
