"""Download the IBM Telco Customer Churn dataset (cached; reproducible)."""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = os.path.join(os.path.dirname(__file__), "Telco-Customer-Churn.csv")


def main() -> None:
    if os.path.exists(OUT):
        print(f"Already cached: {OUT}")
        return
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(OUT, "wb") as f:
        f.write(r.read())
    print(f"Downloaded: {OUT}")


if __name__ == "__main__":
    main()
