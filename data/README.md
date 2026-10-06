# Data

## Source (real data)
- **Dataset:** IBM Telco Customer Churn
- **URL:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- **Accessed:** 2026-10-06
- **Rows:** 7,043 customers × 21 columns (after header)
- **Target:** `Churn` (Yes/No), overall churn rate ≈ 26.5%

## Download
`data/download.py` fetches and caches the CSV (skips if already present):
```
.venv/bin/python data/download.py
```
The CSV is committed to the repo for reproducibility (7 KB-class file is tiny;
no git-ignore needed). `data/clean/` is reserved for processed artifacts and is git-ignored.

## Known quirks (handled in `src/churn/features.py`)
- `TotalCharges` is a string column with 11 blank values (tenure-0 customers) → coerced to NaN, median-imputed.
- Service columns use `"No phone service"` / `"No internet service"` instead of `"No"` → normalized.
- `customerID` is dropped (identifier, not a feature).
