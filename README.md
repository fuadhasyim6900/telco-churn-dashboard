# Telco Churn Dashboard

📉 **Telecom Customer Churn Dashboard** built with **Streamlit** for interactive visualization and analysis of customer churn, segmentation, and retention strategies.

---

## Overview

This project helps a telecom company to:

- Analyze customer churn patterns.
- Identify high-risk and high-value customer segments.
- Visualize churn drivers and revenue loss.
- Support retention strategies through actionable insights.

The dashboard includes:

1. **Project Overview** – Business context, goals, and key metrics.
2. **Churn Dashboard** – KPIs, churn by tenure cohort, contract, and cluster.
3. **Churn Driver Analysis** – Wordcloud for churn reasons, service impact, and satisfaction analysis.
4. **Segmentation & Cohort Analysis** – Cluster profiling, cohort churn heatmap, and revenue lost.

---

## Dataset

The dashboard uses the `customer_churn_dashboard.csv` dataset, which contains:

- `customer_id`: Unique customer identifier.
- `tenure`: Customer tenure in months.
- `monthly_charges`: Monthly subscription fees.
- `total_revenue`: Total revenue contributed by the customer.
- `cltv`: Customer Lifetime Value.
- `churn_value`: Binary indicator of churn (1 = churned, 0 = active).
- `contract`, `tenure_cohort`, `cluster`, and other features for analysis.

> **Note:** Replace this dataset with your own if needed.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/telco-churn-dashboard.git
cd telco-churn-dashboard

Install dependencies:

pip install -r requirements.txt


Run the Streamlit app locally:

streamlit run app.py

Deployment

The app can be deployed to Streamlit Cloud
:

Push your repository to GitHub.

Log in to Streamlit Cloud.

Click New App, select your repository, branch, and app.py.

Click Deploy.

Folder Structure
telco-churn-dashboard/
│
├─ app.py                  # Main Streamlit app
├─ customer_churn_dashboard.csv
├─ requirements.txt
└─ data/                   # Optional: CSV files

License

MIT License

Author

Fuad Hasyim – Data Analyst / Data Scientist
