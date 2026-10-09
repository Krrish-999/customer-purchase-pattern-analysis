"""Calculates the REAL statistics of supermarket_transactions.csv.
Run:  python dataset/validate_dataset.py"""
import os, sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "supermarket_transactions.csv")
df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
req = ["TransactionID","CustomerID","Date","ProductID","Product","Category","Quantity","UnitPrice","TotalAmount"]
missing = [c for c in req if c not in df.columns]
print("File:", path)
print("Missing columns:", missing or "none")
print("Rows (line items):      ", f"{len(df):,}")
print("Transactions:           ", f"{df.TransactionID.nunique():,}")
print("Unique customers:       ", f"{df.CustomerID.nunique():,}")
print("Unique products:        ", f"{df.Product.nunique():,}")
print("Unique categories:      ", f"{df.Category.nunique():,}")
print("Items sold (sum qty):   ", f"{int(df.Quantity.sum()):,}")
print("Avg basket (products/tx):", round(df.groupby('TransactionID').Product.nunique().mean(), 2))
print("Total revenue:          ", f"{df.TotalAmount.sum():,.2f}")
print("Date range:             ", df.Date.min().date(), "->", df.Date.max().date())
print("Null values:", int(df.isna().sum().sum()), "| Duplicate rows:", int(df.duplicated().sum()))
print("TotalAmount == Quantity*UnitPrice for all rows:", bool(((df.Quantity * df.UnitPrice).round(2) - df.TotalAmount).abs().max() < 0.011))
print("\nTransactions per year:\n", df.drop_duplicates("TransactionID").groupby(df.Date.dt.year).size().to_string())
print("\nTop 15 products (by number of transactions):")
print(df.groupby("Product").TransactionID.nunique().sort_values(ascending=False).head(15).to_string())
