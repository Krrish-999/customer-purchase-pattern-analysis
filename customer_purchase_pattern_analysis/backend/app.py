"""Customer Purchase Pattern Analysis in Supermarket - Flask backend.
Run:  python backend/app.py   ->  http://127.0.0.1:5000
All numbers returned by the API are computed from the active dataset (SQLite DB built from the CSV)."""
import io, os, re, json, time, sqlite3, inspect, itertools, sys
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))
from ml_preprocessing import get_dataset_profile, apply_preprocessing_step
from ml_classification import run_classification
from ml_regression import run_regression
from ml_clustering import run_clustering
from ml_comparison import (
    compare_classification, compare_regression, compare_clustering,
    analyze_association_rules, analyze_preprocessing, synthesize_overall_comparison
)

try:
    from mlxtend.frequent_patterns import apriori as mlx_apriori, association_rules as mlx_rules
    HAS_MLXTEND = True
except Exception:  # mlxtend missing -> built-in Apriori keeps the app usable
    HAS_MLXTEND = False

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_DEFAULT = os.path.join(BASE, "dataset", "supermarket_transactions.csv")
DB_PATH = os.path.join(BASE, "database", "supermarket.db")
FRONTEND = os.path.join(BASE, "frontend")
app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
STATE = {
    "df": None,
    "source": None,
    "prep": {},
    "rules": None,
    "prep_df": None,
    "prep_history": [],
    "initial_profile": None
}

# ---------------------------------------------------------------- helpers
def sanitize(o):
    if isinstance(o, dict): return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray): return [sanitize(v) for v in o.tolist()]
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (pd.Timestamp,)): return o.strftime("%Y-%m-%d %H:%M:%S") if (o.hour or o.minute or o.second) else o.strftime("%Y-%m-%d")
    if o is pd.NaT: return None
    return o

def ok(data, code=200): return jsonify(sanitize(data)), code
def err(msg, code=400): return jsonify({"error": msg}), code

ALIASES = {
    "TransactionID": ["transactionid", "transaction", "invoiceno", "invoice", "orderid", "billno", "basketid", "tid", "transid"],
    "CustomerID": ["customerid", "custid", "customer", "clientid", "userid"],
    "Date": ["date", "invoicedate", "transactiondate", "orderdate", "purchasedate", "datetime", "timestamp"],
    "ProductID": ["productid", "stockcode", "sku", "itemid"],
    "Product": ["product", "productname", "item", "itemname", "description", "itemdescription"],
    "Category": ["category", "productcategory", "department", "aisle", "itemcategory"],
    "Quantity": ["quantity", "qty", "units"],
    "UnitPrice": ["unitprice", "price", "priceperunit", "itemprice"],
    "TotalAmount": ["totalamount", "total", "amount", "sales", "revenue", "linetotal"],
}

# ---------------------------------------------------------------- preprocessing
def preprocess(raw):
    """Normalise column names, clean the data, return (clean_df, summary)."""
    notes, derived = [], []
    summary = {"raw_rows": int(len(raw)), "raw_columns": [str(c) for c in raw.columns]}
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in raw.columns}
    ren = {}
    for std, al in ALIASES.items():
        for a in al:
            if a in norm: ren[norm[a]] = std; break
    df = raw.rename(columns=ren)
    df = df[[c for c in ALIASES if c in df.columns]].copy()
    if "TransactionID" not in df or "Product" not in df:
        raise ValueError("Dataset needs at least a transaction id column and a product/item column "
                         "(e.g. TransactionID, Product). Found: " + ", ".join(map(str, raw.columns)))
    for c in ["TransactionID", "Product"]:
        n0 = len(df); df = df[df[c].notna() & (df[c].astype(str).str.strip() != "")]
    summary["missing_removed"] = int(summary["raw_rows"] - len(df))
    for c in ["TransactionID", "CustomerID", "ProductID", "Product", "Category"]:
        if c in df: df[c] = df[c].astype(str).str.strip()
    if "CustomerID" not in df: df["CustomerID"] = "Unknown"; derived.append("CustomerID"); notes.append("No customer column found - all rows assigned to 'Unknown'.")
    if "Category" not in df: df["Category"] = "Uncategorized"; derived.append("Category"); notes.append("No category column found - 'Uncategorized' used.")
    df["Category"] = df["Category"].replace({"nan": "Uncategorized", "": "Uncategorized"})
    if "Date" in df:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad = int(df["Date"].isna().sum())
        if bad: notes.append(f"{bad:,} rows have an unreadable date (kept, excluded from date-based charts).")
    else:
        df["Date"] = pd.NaT; derived.append("Date"); notes.append("No date column found - Yearly Sales are unavailable.")
    if "Quantity" in df: df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    else: df["Quantity"] = 1; derived.append("Quantity"); notes.append("No quantity column found - quantity = 1 assumed.")
    if "UnitPrice" in df: df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0)
    else: df["UnitPrice"] = 0.0; derived.append("UnitPrice"); notes.append("No price column found - revenue figures will be 0.")
    n0 = len(df); df = df[df["Quantity"].notna() & (df["Quantity"] > 0)]; summary["invalid_quantity_removed"] = int(n0 - len(df))
    if "TotalAmount" in df: df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")
    else: df["TotalAmount"] = np.nan; derived.append("TotalAmount")
    df["TotalAmount"] = df["TotalAmount"].fillna((df["Quantity"] * df["UnitPrice"]).round(2))
    if "ProductID" not in df or df["ProductID"].isna().all():
        df["ProductID"] = pd.factorize(df["Product"])[0]; df["ProductID"] = df["ProductID"].map(lambda i: "P%04d" % (i + 1)); derived.append("ProductID")
    df["ProductID"] = df.groupby("Product")["ProductID"].transform("first")
    df["Category"] = df.groupby("Product")["Category"].transform("first")
    n0 = len(df); df = df.drop_duplicates(); summary["duplicates_removed"] = int(n0 - len(df))
    df = df.sort_values(["TransactionID"], kind="stable").reset_index(drop=True)
    summary.update(final_rows=int(len(df)), derived_columns=derived, notes=notes)
    return df, summary

def read_any(name, content):
    ext = os.path.splitext(name.lower())[1]
    if ext in (".xlsx", ".xls"): return pd.read_excel(io.BytesIO(content))
    if ext == ".csv":
        try: return pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError: return pd.read_csv(io.BytesIO(content), encoding="latin-1")
    raise ValueError("Unsupported file type. Upload .csv, .xlsx or .xls")

# ---------------------------------------------------------------- SQLite
SCHEMA = """
DROP TABLE IF EXISTS transaction_items; DROP TABLE IF EXISTS transactions; DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories; DROP TABLE IF EXISTS customers; DROP TABLE IF EXISTS meta;
CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL);
CREATE TABLE customers(customer_id TEXT PRIMARY KEY);
CREATE TABLE products(product_id TEXT PRIMARY KEY, product_name TEXT NOT NULL, category_id INTEGER REFERENCES categories(category_id), avg_unit_price REAL);
CREATE TABLE transactions(transaction_id TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id), txn_date TEXT, total_amount REAL, distinct_products INTEGER);
CREATE TABLE transaction_items(item_id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT REFERENCES transactions(transaction_id), product_id TEXT REFERENCES products(product_id), quantity REAL, unit_price REAL, total_amount REAL);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_items_tx ON transaction_items(transaction_id); CREATE INDEX idx_tx_date ON transactions(txn_date);
"""

def build_db(df, source, prep):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH); con.executescript(SCHEMA)
    cats = pd.DataFrame({"category_name": sorted(df.Category.unique())}); cats["category_id"] = range(1, len(cats) + 1)
    cats[["category_id", "category_name"]].to_sql("categories", con, if_exists="append", index=False)
    cmap = dict(zip(cats.category_name, cats.category_id))
    pd.DataFrame({"customer_id": sorted(df.CustomerID.unique())}).to_sql("customers", con, if_exists="append", index=False)
    p = df.groupby("ProductID").agg(product_name=("Product", "first"), Category=("Category", "first"), avg_unit_price=("UnitPrice", "mean")).reset_index()
    p["category_id"] = p.Category.map(cmap)
    p.rename(columns={"ProductID": "product_id"})[["product_id", "product_name", "category_id", "avg_unit_price"]].to_sql("products", con, if_exists="append", index=False)
    g = df.groupby("TransactionID")
    t = pd.DataFrame({"customer_id": g.CustomerID.first(), "txn_date": g.Date.first().dt.strftime("%Y-%m-%d"),
                      "total_amount": g.TotalAmount.sum().round(2), "distinct_products": g.Product.nunique()}).reset_index().rename(columns={"TransactionID": "transaction_id"})
    t.to_sql("transactions", con, if_exists="append", index=False)
    df.rename(columns={"TransactionID": "transaction_id", "ProductID": "product_id", "Quantity": "quantity", "UnitPrice": "unit_price", "TotalAmount": "total_amount"})[
        ["transaction_id", "product_id", "quantity", "unit_price", "total_amount"]].to_sql("transaction_items", con, if_exists="append", index=False)
    con.executemany("INSERT INTO meta VALUES(?,?)", [("source", source), ("prep", json.dumps(sanitize(prep)))])
    con.commit(); con.close()

def load_from_db():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""SELECT ti.transaction_id AS TransactionID, t.customer_id AS CustomerID, t.txn_date AS Date,
        ti.product_id AS ProductID, p.product_name AS Product, c.category_name AS Category, ti.quantity AS Quantity,
        ti.unit_price AS UnitPrice, ti.total_amount AS TotalAmount
        FROM transaction_items ti JOIN transactions t ON t.transaction_id=ti.transaction_id
        JOIN products p ON p.product_id=ti.product_id JOIN categories c ON c.category_id=p.category_id ORDER BY ti.item_id""", con)
    meta = dict(con.execute("SELECT key,value FROM meta").fetchall()); con.close()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    init_prof = get_dataset_profile(df)
    STATE.update(
        df=df,
        source=meta.get("source", "unknown"),
        prep=json.loads(meta.get("prep", "{}")),
        rules=None,
        prep_df=df.copy(),
        prep_history=[],
        initial_profile=init_prof
    )

def import_dataframe(raw, source):
    clean, prep = preprocess(raw)
    if clean.empty: raise ValueError("No valid rows left after preprocessing.")
    build_db(clean, source, prep); load_from_db()

def init():
    ready = False
    if os.path.exists(DB_PATH):
        try:
            con = sqlite3.connect(DB_PATH); ready = con.execute("SELECT COUNT(*) FROM transaction_items").fetchone()[0] > 0; con.close()
        except Exception: ready = False
    if not ready:
        print("Initialising database from", CSV_DEFAULT)
        import_dataframe(pd.read_csv(CSV_DEFAULT), "supermarket_transactions.csv (default)")
    else:
        load_from_db()
    print(f"Dataset loaded: {STATE['source']} | {len(STATE['df']):,} rows | engine={'mlxtend' if HAS_MLXTEND else 'built-in fallback'}")

# ---------------------------------------------------------------- analytics
def basket_sizes(df): return df.groupby("TransactionID")["Product"].nunique()
def has_dates(df): return df["Date"].notna().any()

def dataset_stats(df):
    s = basket_sizes(df)
    dr = [df.Date.min(), df.Date.max()] if has_dates(df) else [None, None]
    return {"rows": len(df), "columns": len(df.columns), "transactions": int(s.size), "customers": int(df.CustomerID.nunique()),
            "products": int(df.Product.nunique()), "categories": int(df.Category.nunique()), "items_sold": int(round(df.Quantity.sum())),
            "avg_basket": round(float(s.mean()), 2), "revenue": round(float(df.TotalAmount.sum()), 2),
            "date_from": dr[0], "date_to": dr[1]}

def overview():
    df = STATE["df"]; sizes = basket_sizes(df)
    top = df.groupby("Product")["TransactionID"].nunique().sort_values(ascending=False).head(15)
    cap = int(min(max(sizes.max(), 1), 12)); vc = sizes.clip(upper=cap).value_counts().sort_index()
    labels = [f"{i} item{'s' if i > 1 else ''}" if i < cap or sizes.max() <= cap else f"{cap}+ items" for i in range(1, cap + 1)]
    sizes_d = {"labels": labels, "values": [int(vc.get(i, 0)) for i in range(1, cap + 1)]}
    daily = {"labels": [], "values": []}
    if has_dates(df):
        d = df.drop_duplicates("TransactionID")["Date"].dropna().dt.strftime("%Y-%m-%d").value_counts().sort_index().tail(60)
        daily = {"labels": list(d.index), "values": [int(v) for v in d.values]}
    freq = df.groupby("CustomerID")["TransactionID"].nunique()
    bins = [(1, 5, "1-5 purchases"), (6, 10, "6-10 purchases"), (11, 15, "11-15 purchases"), (16, 20, "16-20 purchases"), (21, 10**9, "21+ purchases")]
    fl, fv = [], []
    for lo, hi, lab in bins:
        n = int(((freq >= lo) & (freq <= hi)).sum())
        if n: fl.append(lab); fv.append(n)
    return {"kpis": dataset_stats(df), "top_items": {"labels": list(top.index), "values": [int(v) for v in top.values]},
            "size_distribution": sizes_d, "daily": daily, "frequency": {"labels": fl, "values": fv}}

def yearly():
    df = STATE["df"]
    if not has_dates(df):
        return {"available": False, "message": "The active dataset has no usable Date column, so yearly analysis cannot be calculated."}
    d = df[df.Date.notna()].copy(); d["Year"] = d.Date.dt.year; d["Month"] = d.Date.dt.month; d["Quarter"] = d.Date.dt.quarter
    years = sorted(int(y) for y in d.Year.unique())
    items = {y: int(round(d[d.Year == y].Quantity.sum())) for y in years}
    tx = {y: int(d[d.Year == y].TransactionID.nunique()) for y in years}
    rev = {y: round(float(d[d.Year == y].TotalAmount.sum()), 2) for y in years}
    pct = lambda a, b: None if not a else round((b - a) / a * 100, 2)
    yoy = [{"from": a, "to": b, "items": pct(items[a], items[b]), "transactions": pct(tx[a], tx[b]), "revenue": pct(rev[a], rev[b])} for a, b in zip(years, years[1:])]
    monthly = {y: [int(round(d[(d.Year == y) & (d.Month == m)].Quantity.sum())) for m in range(1, 13)] for y in years}
    quarterly = {y: [int(round(d[(d.Year == y) & (d.Quarter == q)].Quantity.sum())) for q in range(1, 5)] for y in years}
    topn = d.groupby("Product").Quantity.sum().sort_values(ascending=False).head(10).index.tolist()
    piv = d[d.Product.isin(topn)].groupby(["Product", "Year"]).Quantity.sum().unstack(fill_value=0)
    top_items = {"labels": topn, "series": {y: [int(round(piv.loc[p, y])) if y in piv.columns else 0 for p in topn] for y in years}}
    return {"available": True, "years": years, "items_by_year": items, "transactions_by_year": tx, "revenue_by_year": rev, "yoy": yoy,
            "monthly": monthly, "quarterly": quarterly, "top_items": top_items,
            "note": "Only one year found - year-over-year growth needs at least two years." if len(years) < 2 else ""}

# ---------------------------------------------------------------- Apriori
def apriori_fallback(basket, min_support, max_len):
    XT = np.ascontiguousarray(basket.values.T); n = XT.shape[1]; cols = list(basket.columns)
    sup1 = XT.sum(1) / n; freq = {(int(i),): float(sup1[i]) for i in np.where(sup1 >= min_support)[0]}
    allf = dict(freq); cur = sorted(freq); k = 1
    while cur and k < max_len:
        groups = {}
        for c in cur: groups.setdefault(c[:-1], []).append(c[-1])
        curset = set(cur); nxt = {}
        for pre, lasts in groups.items():
            for a, b in itertools.combinations(sorted(lasts), 2):
                cand = pre + (a, b)
                if all(cand[:i] + cand[i + 1:] in curset for i in range(len(cand))):
                    m = XT[cand[0]].copy()
                    for j in cand[1:]: m &= XT[j]
                    s = m.sum() / n
                    if s >= min_support: nxt[cand] = float(s)
        allf.update(nxt); cur = sorted(nxt); k += 1
    return pd.DataFrame({"support": list(allf.values()), "itemsets": [frozenset(cols[i] for i in t) for t in allf]})

def rules_fallback(freq, min_threshold):
    sup = {fs: s for fs, s in zip(freq["itemsets"], freq["support"])}; rows = []
    for fs, s in sup.items():
        if len(fs) < 2: continue
        for r in range(1, len(fs)):
            for a in itertools.combinations(sorted(fs), r):
                a = frozenset(a); c = fs - a; conf = s / sup[a]; lift = conf / sup[c]
                if lift >= min_threshold:
                    rows.append((a, c, sup[a], sup[c], s, conf, lift, s - sup[a] * sup[c], (1 - sup[c]) / (1 - conf) if conf < 1 else np.inf))
    return pd.DataFrame(rows, columns=["antecedents", "consequents", "antecedent support", "consequent support", "support", "confidence", "lift", "leverage", "conviction"])

def run_apriori(min_support, min_conf, min_lift, level="Product", max_len=3):
    df = STATE["df"]; t0 = time.time(); steps = []
    col = "Category" if level == "Category" else "Product"
    steps.append(f"Loaded {len(df):,} line items from the active dataset (SQLite).")
    pairs = df[["TransactionID", col]].drop_duplicates(); n_dup = len(df) - len(pairs)
    steps.append(f"Grouped line items into transactions at {col} level; removed {n_dup:,} repeated {col.lower()} entries inside the same transaction.")
    basket = pd.crosstab(pairs["TransactionID"], pairs[col]).gt(0)
    n_tx, n_items = basket.shape
    if n_items > 1500:
        keep = basket.sum().sort_values(ascending=False).head(1500).index; basket = basket[keep]
        steps.append(f"Kept the 1,500 most frequent of {n_items:,} items to stay within memory.")
    steps.append(f"Built a boolean basket matrix: {n_tx:,} transactions x {basket.shape[1]:,} items.")
    t1 = time.time()
    if HAS_MLXTEND:
        freq = mlx_apriori(basket, min_support=min_support, use_colnames=True, max_len=max_len); engine = "mlxtend Apriori"
    else:
        freq = apriori_fallback(basket, min_support, max_len); engine = "built-in Apriori (mlxtend not installed)"
    rules = pd.DataFrame()
    if len(freq) and (freq["itemsets"].map(len) >= 2).any():
        if HAS_MLXTEND:
            kw = {"metric": "lift", "min_threshold": min_lift}
            if "num_itemsets" in inspect.signature(mlx_rules).parameters: kw["num_itemsets"] = n_tx
            rules = mlx_rules(freq, **kw)
        else:
            rules = rules_fallback(freq, min_lift)
        rules = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)].sort_values(["lift", "confidence"], ascending=False)
    t2 = time.time()
    steps.append(f"Apriori found {len(freq):,} frequent itemsets at min support {min_support:.3f}.")
    steps.append(f"Generated {len(rules):,} association rules with confidence >= {min_conf:.2f} and lift >= {min_lift:.2f}.")
    its = freq.assign(size=freq["itemsets"].map(len)).sort_values(["support"], ascending=False)
    itemsets = [{"items": sorted(r.itemsets), "support": float(r.support), "size": int(r.size)} for r in its.head(300).itertuples()]
    rl = [{"antecedents": sorted(r.antecedents), "consequents": sorted(r.consequents), "support": float(r.support), "confidence": float(r.confidence),
           "lift": float(r.lift), "leverage": float(r.leverage), "conviction": None if np.isinf(r.conviction) else float(r.conviction)} for r in rules.head(300).itertuples()]
    res = {"engine": engine, "level": level, "params": {"min_support": min_support, "min_confidence": min_conf, "min_lift": min_lift, "max_len": max_len},
           "n_transactions": int(n_tx), "n_items": int(basket.shape[1]), "n_itemsets": int(len(freq)), "n_rules": int(len(rules)),
           "itemsets_by_size": {int(k): int(v) for k, v in its["size"].value_counts().sort_index().items()},
           "itemsets": itemsets, "rules": rl, "steps": steps,
           "timing": {"preprocess_s": round(t1 - t0, 2), "apriori_s": round(t2 - t1, 2)}}
    STATE["rules"] = res; return res

DEFAULTS = {"min_support": 0.005, "min_confidence": 0.25, "min_lift": 1.20, "level": "Product", "max_len": 3}
def ensure_rules():
    if STATE["rules"] is None: run_apriori(DEFAULTS["min_support"], DEFAULTS["min_confidence"], DEFAULTS["min_lift"], DEFAULTS["level"], DEFAULTS["max_len"])
    return STATE["rules"]

def fmt(items): return " + ".join(items)
def interpretation(res):
    r = res["rules"]
    if not r: return ["No rules satisfy the current thresholds. Lower the minimum support, confidence or lift and run Apriori again."]
    top = max(r, key=lambda x: x["lift"]); conf = max(r, key=lambda x: x["confidence"]); sup = max(r, key=lambda x: x["support"])
    pct = lambda v: f"{v*100:.1f}%"
    return [f"{res['n_rules']:,} rules were mined from {res['n_transactions']:,} transactions ({res['level'].lower()} level, engine: {res['engine']}).",
            f"Strongest association (lift {top['lift']:.2f}): {fmt(top['antecedents'])} -> {fmt(top['consequents'])}. Baskets with the first part contain the second part {top['lift']:.1f}x more often than average ({pct(top['confidence'])} of the time).",
            f"Most reliable rule (confidence {pct(conf['confidence'])}): {fmt(conf['antecedents'])} -> {fmt(conf['consequents'])}.",
            f"Most frequent pairing (support {pct(sup['support'])}): {fmt(sup['antecedents'])} -> {fmt(sup['consequents'])}, appearing in about {int(round(sup['support']*res['n_transactions'])):,} transactions."]

THEMES = [("purple", "Bundle offer", "Create a combo price for {a} + {c}; the pair is bought together far above chance."),
          ("green", "Shelf placement", "Place {c} next to {a} (or add a cross-aisle sign) to capture the follow-on purchase."),
          ("orange", "Targeted promotion", "Use a coupon on {c} for shoppers who put {a} in the basket to lift basket value."),
          ("pink", "Recommendation", "Suggest {c} at checkout / in the app whenever {a} is added to the cart.")]
def insights():
    res = ensure_rules(); seen, cards = set(), []
    for r in sorted(res["rules"], key=lambda x: (x["lift"], x["confidence"]), reverse=True):
        key = frozenset(r["antecedents"] + r["consequents"])
        if key in seen: continue
        seen.add(key); i = len(cards); color, kind, tip = THEMES[i % 4]
        a, c = fmt(r["antecedents"]), fmt(r["consequents"])
        cards.append({"color": color, "type": kind, "title": f"{a} -> {c}", "metrics": {"support": r["support"], "confidence": r["confidence"], "lift": r["lift"]},
                      "text": f"{r['confidence']*100:.0f}% of baskets with {a} also contain {c} (lift {r['lift']:.2f}, support {r['support']*100:.1f}%). " + tip.format(a=a, c=c)})
        if len(cards) == 8: break
    df = STATE["df"]; extra = []
    if has_dates(df):
        d = df.drop_duplicates("TransactionID").dropna(subset=["Date"])
        wd = d.Date.dt.day_name().value_counts(); mo = d.Date.dt.month_name().value_counts()
        extra.append({"color": "green", "type": "Sales data", "title": f"Busiest day: {wd.index[0]}", "text": f"{wd.iloc[0]:,} transactions fall on {wd.index[0]}s and the busiest month is {mo.index[0]} ({mo.iloc[0]:,} transactions). Schedule staff and fresh-stock deliveries ahead of these peaks."})
    cr = df.groupby("Category").TotalAmount.sum().sort_values(ascending=False)
    extra.append({"color": "purple", "type": "Sales data", "title": f"Top revenue category: {cr.index[0]}", "text": f"{cr.index[0]} generates {cr.iloc[0]/cr.sum()*100:.1f}% of total revenue, followed by {cr.index[1] if len(cr) > 1 else '-'}. Keep these categories well stocked and prominently placed."})
    return {"cards": cards, "extra": extra, "summary": interpretation(res)}

ALGOS = {
 "implemented": "Apriori",
 "algorithms": [
  {"name": "Apriori", "status": "IMPLEMENTED", "full": "Apriori frequent-itemset mining (Agrawal & Srikant, 1994)",
   "concept": "Level-wise (breadth-first) search that finds all itemsets whose support is at least the minimum support, using the property that every subset of a frequent itemset must also be frequent.",
   "candidate_generation": "Joins frequent (k-1)-itemsets to build candidate k-itemsets.",
   "pruning": "Apriori property: a candidate is discarded if any of its (k-1)-subsets is infrequent; remaining candidates are counted with a database scan.",
   "purpose": "Discover frequent item combinations and generate association rules (support, confidence, lift) for market-basket analysis.",
   "strengths": "Simple, easy to explain, widely implemented (mlxtend).", "limitations": "Many candidates and repeated database scans when support is low or items are many."},
  {"name": "DHP", "status": "CONCEPTUAL COMPARISON", "full": "Direct Hashing and Pruning (Park, Chen & Yu, 1995)",
   "concept": "An Apriori variant that uses a hash table built while counting k-itemsets to filter candidates of the next level, and trims the transaction database as it goes.",
   "candidate_generation": "While counting 1-itemsets, all 2-itemsets of each transaction are hashed into buckets with counters.",
   "pruning": "A candidate 2-itemset is kept only if its bucket count reaches minimum support; transactions/items that cannot contribute are removed from later passes.",
   "purpose": "Reduce the size of the candidate set (especially C2) and the database scanned, making Apriori-style mining faster.",
   "strengths": "Much smaller candidate sets in early passes; shrinking database.", "limitations": "Extra memory/time for the hash table; hash collisions reduce pruning power."},
  {"name": "CHARM", "status": "CONCEPTUAL COMPARISON", "full": "Closed Association Rule Mining (Zaki & Hsiao, 2002)",
   "concept": "Mines only closed frequent itemsets (frequent itemsets with no superset of equal support) using an itemset-tidset search tree, giving a compact lossless representation.",
   "candidate_generation": "Depth-first extension of itemset-tidset pairs; tidset intersections give support directly (vertical format).",
   "pruning": "Subsumed or non-closed itemsets are skipped by comparing tidsets (equal / subset properties) plus hash-based closure checking.",
   "purpose": "Avoid redundant itemsets and rules by keeping only closed itemsets while retaining all support information.",
   "strengths": "Far fewer itemsets on dense data; no repeated database scans.", "limitations": "Tidset memory use; more complex to implement and explain."}],
 "comparison": [
  {"aspect": "Search strategy", "values": ["Breadth-first, level-wise", "Breadth-first with hash filtering", "Depth-first on itemset-tidset tree"]},
  {"aspect": "Data layout", "values": ["Horizontal (transactions)", "Horizontal (shrinking)", "Vertical (item -> tidset)"]},
  {"aspect": "Output", "values": ["All frequent itemsets", "All frequent itemsets", "Closed frequent itemsets only"]},
  {"aspect": "Main idea", "values": ["Subset-based pruning", "Hash-bucket candidate pruning", "Closure checking via tidsets"]},
  {"aspect": "Database scans", "values": ["One per level", "One per level (smaller db)", "Few / none after vertical conversion"]},
  {"aspect": "Status in this project", "values": ["Implemented & executed on the dataset", "Concept only - not executed", "Concept only - not executed"]}],
 "disclaimer": "Only Apriori is executed in this project. DHP and CHARM are described conceptually; no execution times, memory figures or rule counts are shown for them because none were measured."}

# ---------------------------------------------------------------- routes
@app.route("/")
def index(): return send_from_directory(FRONTEND, "index.html")

@app.route("/api/overview")
def api_overview(): return ok(overview())

@app.route("/api/yearly")
def api_yearly(): return ok(yearly())

@app.route("/api/algorithms")
def api_algorithms(): return ok(ALGOS)

@app.route("/api/config")
def api_config(): return ok({"defaults": DEFAULTS, "engine": "mlxtend" if HAS_MLXTEND else "built-in fallback"})

@app.route("/api/dataset")
def api_dataset():
    df = STATE["df"]; prev = df.head(25).copy(); prev["Date"] = prev["Date"].dt.strftime("%Y-%m-%d")
    return ok({"source": STATE["source"], "stats": dataset_stats(df), "columns": list(df.columns), "preview": prev.where(prev.notna(), None).to_dict("records"), "preprocessing": STATE["prep"]})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename: return err("No file received.")
    try:
        import_dataframe(read_any(f.filename, f.read()), f.filename)
    except Exception as e:
        return err(str(e))
    return api_dataset()

@app.route("/api/reset", methods=["POST"])
def api_reset():
    import_dataframe(pd.read_csv(CSV_DEFAULT), "supermarket_transactions.csv (default)"); return api_dataset()

@app.route("/api/apriori", methods=["POST"])
def api_apriori():
    b = request.get_json(silent=True) or {}
    try:
        s = float(b.get("min_support", DEFAULTS["min_support"])); c = float(b.get("min_confidence", DEFAULTS["min_confidence"])); l = float(b.get("min_lift", DEFAULTS["min_lift"]))
        ml = int(b.get("max_len", 3)); lvl = "Category" if b.get("level") == "Category" else "Product"
    except (TypeError, ValueError): return err("Invalid parameters.")
    if not (0 < s <= 1 and 0 <= c <= 1 and l >= 0 and 2 <= ml <= 5): return err("Support must be in (0,1], confidence in [0,1], lift >= 0, max length 2-5.")
    if s < 0.002: return err("Minimum support below 0.002 would take too long on a typical computer. Use 0.002 or higher.")
    return ok(run_apriori(s, c, l, lvl, ml))

@app.route("/api/rules")
def api_rules():
    res = ensure_rules(); return ok({"result": res, "interpretation": interpretation(res)})

@app.route("/api/insights")
def api_insights(): return ok(insights())

# ---------------------------------------------------------------- Data Mining routes

@app.route("/api/dm/columns")
def api_dm_columns():
    df = STATE.get("prep_df") if STATE.get("prep_df") is not None else STATE.get("df")
    if df is None: return err("No dataset loaded.")
    prof = get_dataset_profile(df)
    return ok({"columns": prof["columns_info"], "names": prof["column_names"], "rows": prof["rows"]})

@app.route("/api/preprocessing/status")
def api_preprocessing_status():
    df_orig = STATE.get("df")
    if df_orig is None: return err("No dataset loaded.")
    df_prep = STATE.get("prep_df")
    if df_prep is None:
        df_prep = df_orig.copy()
        STATE["prep_df"] = df_prep
        STATE["prep_history"] = []

    init_prof = STATE.get("initial_profile")
    if not init_prof:
        init_prof = get_dataset_profile(df_orig)
        STATE["initial_profile"] = init_prof

    curr_prof = get_dataset_profile(df_prep)

    prev_before = df_orig.head(15).copy()
    if "Date" in prev_before.columns and pd.api.types.is_datetime64_any_dtype(prev_before["Date"]):
        prev_before["Date"] = prev_before["Date"].dt.strftime("%Y-%m-%d")
        
    prev_after = df_prep.head(15).copy()
    if "Date" in prev_after.columns and pd.api.types.is_datetime64_any_dtype(prev_after["Date"]):
        prev_after["Date"] = prev_after["Date"].dt.strftime("%Y-%m-%d")

    return ok({
        "before": init_prof,
        "after": curr_prof,
        "history": STATE.get("prep_history", []),
        "preview_before": prev_before.where(prev_before.notna(), None).to_dict("records"),
        "preview_after": prev_after.where(prev_after.notna(), None).to_dict("records")
    })

@app.route("/api/preprocessing/apply", methods=["POST"])
def api_preprocessing_apply():
    body = request.get_json(silent=True) or {}
    op = body.get("operation")
    params = body.get("params", {})
    if not op: return err("No operation specified.")

    df_prep = STATE.get("prep_df")
    if df_prep is None:
        df_prep = STATE["df"].copy()
        STATE["prep_df"] = df_prep

    try:
        new_df, msg, changes = apply_preprocessing_step(df_prep, op, params)
        STATE["prep_df"] = new_df
        history = STATE.setdefault("prep_history", [])
        history.append({
            "step": len(history) + 1,
            "operation": op,
            "message": msg,
            "changes": changes,
            "time": time.strftime("%H:%M:%S")
        })
    except Exception as e:
        return err(str(e))

    return api_preprocessing_status()

@app.route("/api/preprocessing/reset", methods=["POST"])
def api_preprocessing_reset():
    if STATE.get("df") is None: return err("No dataset loaded.")
    STATE["prep_df"] = STATE["df"].copy()
    STATE["prep_history"] = []
    STATE["initial_profile"] = get_dataset_profile(STATE["df"])
    return api_preprocessing_status()

@app.route("/api/preprocessing/commit", methods=["POST"])
def api_preprocessing_commit():
    if STATE.get("prep_df") is None: return err("No preprocessed dataset.")
    STATE["df"] = STATE["prep_df"].copy()
    STATE["initial_profile"] = get_dataset_profile(STATE["df"])
    STATE["rules"] = None
    return ok({"status": "success", "message": f"Preprocessed dataset saved as active baseline ({len(STATE['df']):,} rows, {len(STATE['df'].columns)} columns)."})

@app.route("/api/classification/train", methods=["POST"])
def api_classification_train():
    b = request.get_json(silent=True) or {}
    algo = b.get("algorithm", "decision_tree")
    target = b.get("target_column")
    features = b.get("feature_columns", [])
    params = b.get("params", {})
    use_prep = b.get("use_preprocessed", True)

    df_to_use = STATE.get("prep_df") if (use_prep and STATE.get("prep_df") is not None) else STATE.get("df")
    if df_to_use is None: return err("No dataset loaded.")
    if not target: return err("Please select a target column for classification.")
    if not features: return err("Please select at least one feature column.")

    try:
        res = run_classification(df_to_use, algo, target, features, params)
        return ok(res)
    except Exception as e:
        return err(str(e))

@app.route("/api/regression/train", methods=["POST"])
def api_regression_train():
    b = request.get_json(silent=True) or {}
    algo = b.get("algorithm", "linear")
    target = b.get("target_column")
    features = b.get("feature_columns", [])
    params = b.get("params", {})
    use_prep = b.get("use_preprocessed", True)

    df_to_use = STATE.get("prep_df") if (use_prep and STATE.get("prep_df") is not None) else STATE.get("df")
    if df_to_use is None: return err("No dataset loaded.")
    if not target: return err("Please select a target column for regression.")
    if not features: return err("Please select at least one feature column.")

    try:
        res = run_regression(df_to_use, algo, target, features, params)
        return ok(res)
    except Exception as e:
        return err(str(e))

@app.route("/api/clustering/train", methods=["POST"])
def api_clustering_train():
    b = request.get_json(silent=True) or {}
    algo = b.get("algorithm", "kmeans")
    features = b.get("feature_columns", [])
    mode = b.get("mode", "rows")
    params = b.get("params", {})
    use_prep = b.get("use_preprocessed", True)

    df_to_use = STATE.get("prep_df") if (use_prep and STATE.get("prep_df") is not None) else STATE.get("df")
    if df_to_use is None: return err("No dataset loaded.")

    try:
        res = run_clustering(df_to_use, algo, features, mode=mode, params=params)
        return ok(res)
    except Exception as e:
        return err(str(e))

@app.route("/api/comparison/benchmark", methods=["GET", "POST"])
def api_comparison_benchmark():
    body = request.get_json(silent=True) or {} if request.method == "POST" else {}
    use_prep = body.get("use_preprocessed", True)
    df_to_use = STATE.get("prep_df") if (use_prep and STATE.get("prep_df") is not None) else STATE.get("df")
    if df_to_use is None: return err("No dataset loaded.")

    techs = body.get("techniques", {})
    run_all = not bool(techs)

    clf_enabled = techs.get("classification", {}).get("enabled", True) if not run_all else True
    reg_enabled = techs.get("regression", {}).get("enabled", True) if not run_all else True
    clu_enabled = techs.get("clustering", {}).get("enabled", True) if not run_all else True
    rules_enabled = techs.get("rules", {}).get("enabled", True) if not run_all else True
    prep_enabled = techs.get("preprocessing", {}).get("enabled", True) if not run_all else True

    clf_res = None
    reg_res = None
    clu_res = None
    rules_res = None
    prep_res = None

    # Classification
    if clf_enabled:
        cfg = techs.get("classification", {})
        algos = cfg.get("algorithms", ["decision_tree", "random_forest", "knn", "logistic_regression", "naive_bayes"])
        target = cfg.get("target_column", "Category")
        feats = cfg.get("feature_columns", ["Quantity", "UnitPrice", "TotalAmount"])
        params = cfg.get("params", {"sample_size": 2500, "test_size": 0.2})
        clf_res = compare_classification(df_to_use, algos, target, feats, params)

    # Regression
    if reg_enabled:
        cfg = techs.get("regression", {})
        algos = cfg.get("algorithms", ["linear", "polynomial", "decision_tree", "random_forest"])
        target = cfg.get("target_column", "TotalAmount")
        feats = cfg.get("feature_columns", ["Quantity", "UnitPrice"])
        params = cfg.get("params", {"sample_size": 2500, "test_size": 0.2, "poly_degree": 2, "dt_max_depth": 6, "rf_n_estimators": 40})
        reg_res = compare_regression(df_to_use, algos, target, feats, params)

    # Clustering
    if clu_enabled:
        cfg = techs.get("clustering", {})
        algos = cfg.get("algorithms", ["kmeans", "agglomerative", "dbscan"])
        mode = cfg.get("mode", "rows")
        feats = cfg.get("feature_columns", ["Quantity", "UnitPrice", "TotalAmount"] if mode == "rows" else ["Frequency", "Monetary", "Recency"])
        params = cfg.get("params", {"sample_size": 2000, "n_clusters": 3, "scale_features": True})
        clu_res = compare_clustering(df_to_use, algos, feats, mode, params)

    # Association Rules
    if rules_enabled:
        cfg = techs.get("rules", {})
        if "params" in cfg and cfg["params"]:
            p = cfg["params"]
            rules_raw = run_apriori(p.get("min_support", 0.005), p.get("min_confidence", 0.25), p.get("min_lift", 1.2), p.get("level", "Product"), p.get("max_len", 3))
        else:
            rules_raw = ensure_rules()
        rules_res = analyze_association_rules(rules_raw)

    # Preprocessing
    if prep_enabled:
        init_p = STATE.get("initial_profile") or get_dataset_profile(STATE.get("df"))
        curr_p = get_dataset_profile(STATE.get("prep_df") if STATE.get("prep_df") is not None else STATE.get("df"))
        prep_res = analyze_preprocessing(init_p, curr_p, STATE.get("prep_history", []))

    holistic = synthesize_overall_comparison(clf_res, reg_res, clu_res, rules_res, prep_res)

    return ok({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classification": clf_res,
        "regression": reg_res,
        "clustering": clu_res,
        "rules": rules_res,
        "preprocessing": prep_res,
        "holistic": holistic
    })

if __name__ == "__main__":
    init()
    app.run(host="127.0.0.1", port=5000, debug=False)
