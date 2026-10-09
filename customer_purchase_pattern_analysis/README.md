# Supermarket Data Mining Learning & Analysis Platform (DWM PBL)

A complete **Data Mining Techniques Web Application** built with Flask, Pandas, Scikit-Learn, mlxtend, SQLite, and Chart.js.

The platform provides a comprehensive suite of Data Mining techniques:
1. **Data Preprocessing** (Missing value imputation, deduplication, type conversion, categorical encoding, feature scaling, outlier detection/handling, basic cleaning, feature selection, before/after comparison)
2. **Classification** (Decision Tree, KNN, Logistic Regression, Naive Bayes, Random Forest, SVM with Accuracy, Precision, Recall, F1, Confusion Matrix heatmap, Feature Importances, Sample Predictions)
3. **Regression** (Linear, Multiple Linear, Polynomial, Decision Tree, Random Forest with MAE, MSE, RMSE, R² Score, Actual vs Predicted scatter plot, Feature Importances, Sample Predictions)
4. **Clustering** (K-Means with centroids, Agglomerative, DBSCAN with Silhouette Score, Inertia, 2D PCA/feature projection scatter plot, Cluster Profiles, and Customer RFM segmentation)
5. **Association Rule Mining** (Apriori frequent itemsets, support, confidence, lift, interactive scatter plot, rule cards, and conceptual comparison with DHP and CHARM)

## Run
```bash
pip install -r requirements.txt
python backend/app.py
```
Open **http://127.0.0.1:5000** in your browser.

## Structure
```
backend/
  app.py                    Flask API and application routing
  ml_preprocessing.py       Data cleaning, imputation, encoding, scaling, outliers, feature selection
  ml_classification.py      Classification algorithms (DT, KNN, LR, NB, RF, SVM) & evaluation
  ml_regression.py          Regression algorithms (Linear, Multi-Linear, Poly, DT, RF) & metrics
  ml_clustering.py          Clustering algorithms (KMeans, Agglomerative, DBSCAN, RFM customer analysis)
frontend/
  index.html                Complete dashboard & workbenches for all Data Mining techniques
  style.css                 Modern theme, glassmorphism, responsive cards, confusion matrix styling
  script.js                 Client-side controller, API client, charts, interactive workbench state
dataset/
  supermarket_transactions.csv  Default dataset (35,000 transactions, 128,092 line items)
database/
  supermarket.db            SQLite DB (auto-created from the CSV if missing)
```

## Navigation & Workbenches
- **Dashboard**: Overview of application, quick launch cards for each technique, KPI counters, top items, and volume charts.
- **Dataset**: Schema explorer, data types, missing counts, statistics, preview table, and file upload (CSV/XLSX/XLS).
- **Data Preprocessing**: Interactive transformation pipeline with 8 toolsets, before & after metrics, pipeline history, and commit/reset controls.
- **Classification**: Feature and target selector, algorithm picker, hyperparameters, accuracy/precision/recall/F1 metrics, interactive confusion matrix heatmap, feature importance chart, and test prediction preview.
- **Regression**: Continuous numerical prediction, polynomial modeling, MAE/MSE/RMSE/R² metrics, actual vs predicted scatter plot with ideal fit line, and error preview.
- **Clustering**: Unsupervised segmentation at transaction level or Customer RFM level (Recency, Frequency, Monetary), K-Means centroids, Silhouette scores, 2D projection scatter plot, and cluster profile tables.
- **Association Rules**: Complete Apriori frequent itemset mining, support/confidence/lift controls, rules table, scatter plot, and rule cards.
- **Rule Mining Theory**: Side-by-side conceptual comparison: Apriori vs DHP vs CHARM.
- **Business Insights**: Actionable retail strategies generated from mined association rules.
- **Yearly Sales**: Long-term trends and seasonality analysis.

## API Endpoints
- **Dataset & Overview**: `GET /api/overview`, `GET /api/yearly`, `GET /api/dataset`, `GET /api/dm/columns`, `POST /api/upload`, `POST /api/reset`
- **Data Preprocessing**: `GET /api/preprocessing/status`, `POST /api/preprocessing/apply`, `POST /api/preprocessing/reset`, `POST /api/preprocessing/commit`
- **Classification**: `POST /api/classification/train`
- **Regression**: `POST /api/regression/train`
- **Clustering**: `POST /api/clustering/train`
- **Association Rules**: `POST /api/apriori`, `GET /api/rules`, `GET /api/insights`, `GET /api/algorithms`, `GET /api/config`
