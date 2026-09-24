# E-Commerce Customer Intelligence & Sales Analytics

**IBM SkillsBuild Data Analytics with AI Internship 2026**

---

## Project Objective

Analyze historical e-commerce transaction data to understand sales performance, customer behavior, customer retention, product performance, product relationships, customer value, and customer inactivity risk, and derive evidence-based business recommendations.

---

## Dataset

- **Source:** https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii
- **Description:** Online retail transaction data covering two periods across two Excel sheets.
- **Original file:** `online_retail_II.xlsx` — read-only; never modified during analysis.
- **Citation:** Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5CG6D
---

## Project Structure

```
E-Commerce-Customer-Intelligence/
│
├── data/
│   └── online_retail_II.xlsx          # Original dataset (read-only)
│
├── notebooks/
│   └── 01_data_audit.ipynb            # Data audit notebook
│
├── src/                               # Reusable Python modules (future)
│
├── outputs/
│   └── figures/                       # Saved visualizations
│
├── README.md
└── requirements.txt
```

---

## Planned Analysis Stages

1. Data quality assessment
2. Data cleaning and preprocessing
3. Feature engineering
4. Executive KPIs
5. Sales and revenue analytics
6. Geographic analytics
7. Product intelligence
8. Customer intelligence
9. RFM analysis
10. Customer segmentation
11. Cohort and retention analysis
12. Market basket analysis
13. Predictive analytics
14. Model explainability
15. Business insights & strategic recommendations
16. Streamlit dashboard (app.py)
17. Final DOCX Report Generation

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebooks
jupyter notebook PrathameshJadhav_EcommerceCustomerIntelligence.ipynb

# Run the Streamlit Dashboard
streamlit run app.py
```
