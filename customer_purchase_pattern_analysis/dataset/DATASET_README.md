# Dataset: supermarket_transactions.csv

Synthetic (simulated) supermarket sales data created for the DWM PBL project
"Customer Purchase Pattern Analysis in Supermarket". It is **not** real customer data.
It was produced by `generate_dataset.py` (fixed random seed, reproducible) and the finished CSV is included so it opens in Excel.

## Columns
| Column | Meaning |
|---|---|
| TransactionID | Basket / bill number (T00001 ...) - one transaction has several rows |
| CustomerID | Customer (C0001 ... C3000) |
| Date | Purchase date, YYYY-MM-DD (2023-01-01 to 2024-12-31) |
| ProductID | Product code (P0001 ...) |
| Product | Product name with pack size |
| Category | One of 130 supermarket categories |
| Quantity | Units bought (1-4) |
| UnitPrice | Price per unit (generic currency units; 2024 prices ~3.5% higher, ~10% of lines on promotion) |
| TotalAmount | Quantity x UnitPrice |

## Actual size (run `python dataset/validate_dataset.py` to recompute)
35,000 transactions | 128,092 line items (rows) | 3,000 customers | 470 products | 130 categories |
172,611 items sold (sum of Quantity) | 2023-01-01 to 2024-12-31 | 16,739 transactions in 2023, 18,261 in 2024.

## How realism was added
- Customers have personal favourite categories; some households have babies, dogs, cats, drink alcohol, or avoid meat.
- Staples (fruit, milk, bread, cheese, eggs, chicken, rice, pasta, coffee) are bought more often.
- Weekend and December peaks, and ~8% higher traffic in 2024.
- Probabilistic product associations (e.g. pasta -> pasta sauce 50-55%, coffee -> sugar ~45%, cereal -> milk 55-60%,
  chicken -> rice ~35%, tea -> biscuits ~50%). These are probabilities, not rules, so baskets vary.

## Definitions used by the dashboard
- **Transaction** = distinct TransactionID. **Items Sold** = sum of Quantity.
- **Average Basket Size** = average number of *distinct products* per transaction.
- Apriori treats each transaction as a set of distinct products (or categories).
