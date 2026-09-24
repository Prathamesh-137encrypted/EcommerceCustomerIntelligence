"""
Notebook 03 — Feature Engineering & KPIs
Execution script: produces all outputs and validates all metrics.
Run from project root: python src/nb03_runner.py
"""

import os
import sys
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Settings ─────────────────────────────────────────────────────────────────
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.width', 130)
warnings.filterwarnings('ignore', message='.*data validation.*', category=UserWarning, module='openpyxl')

DATA_PATH   = 'data/online_retail_II.xlsx'
FIGURES_DIR = 'outputs/figures/'
OUTPUTS_DIR = 'outputs/'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD AND REBUILD CLEANED DATASETS (reuse NB02 logic)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — Loading and rebuilding cleaned datasets")
print("=" * 70)

xl_file = pd.ExcelFile(DATA_PATH, engine='openpyxl')
sheet_names = xl_file.sheet_names
print(f"Sheets: {sheet_names}")

sheets = {}
for name in sheet_names:
    print(f"  Loading '{name}' ...", end=' ')
    df = pd.read_excel(DATA_PATH, sheet_name=name, engine='openpyxl',
                       dtype={'Invoice': str, 'StockCode': str})
    sheets[name] = df
    print(f"{len(df):,} rows")

tagged = []
for name, df in sheets.items():
    tmp = df.copy()
    tmp['_sheet'] = name
    tagged.append(tmp)

raw = pd.concat(tagged, ignore_index=True)
raw['InvoiceDate'] = pd.to_datetime(raw['InvoiceDate'], errors='coerce')
RAW_ROW_COUNT = len(raw)
print(f"\nRaw rows: {RAW_ROW_COUNT:,}")

# ── Deduplication ─────────────────────────────────────────────────────────────
dup_mask = raw.duplicated(keep='first')
n_duplicates = dup_mask.sum()
cleaned_transactions = raw[~dup_mask].copy().reset_index(drop=True)
assert len(cleaned_transactions) + n_duplicates == RAW_ROW_COUNT
print(f"After dedup: {len(cleaned_transactions):,} rows  (removed {n_duplicates:,})")

# ── Price flags ───────────────────────────────────────────────────────────────
cleaned_transactions['is_zero_price']     = cleaned_transactions['Price'] == 0
cleaned_transactions['is_negative_price'] = cleaned_transactions['Price'] < 0

# ── StockCode classification ─────────────────────────────────────────────────
def classify_stockcode(code, description):
    code = str(code).strip().upper() if pd.notna(code) else ''
    desc = str(description).strip().upper() if pd.notna(description) else ''
    if re.match(r'^TEST', code) or 'TEST' in desc:
        return 'TEST'
    if code in ('B', 'ADJUST2') or 'BAD DEBT' in desc or 'BAD-DEBT' in desc:
        return 'BAD_DEBT'
    if code.startswith('GIFT') or 'GIFT VOUCHER' in desc or 'GIFT_0' in code:
        return 'GIFT_VOUCHER'
    if code in ('POST', 'DOT', 'C2') or 'POSTAGE' in desc or 'CARRIAGE' in desc:
        return 'POSTAGE'
    if code in ('BANK CHARGES', 'BANKCHARGES', 'AMAZONFEE') \
            or 'BANK CHARGE' in desc or 'AMAZON FEE' in desc or 'FEE' in desc:
        return 'FEE_OR_CHARGE'
    if code == 'D' or 'DISCOUNT' in desc:
        return 'DISCOUNT'
    if code == 'S' or 'SAMPLE' in desc:
        return 'SAMPLE'
    if code in ('M', 'ADJUST', 'CRUK') \
            or 'MANUAL' in desc or 'ADJUST' in desc or 'CRUK' in desc:
        return 'MANUAL_ADJUSTMENT'
    if re.match(r'^\d{5}[A-Z]?$', code):
        return 'MERCHANDISE'
    return 'UNCLASSIFIED_NON_STANDARD'

cleaned_transactions['product_type'] = cleaned_transactions.apply(
    lambda r: classify_stockcode(r['StockCode'], r['Description']), axis=1)

# ── Transaction type ──────────────────────────────────────────────────────────
def assign_transaction_type(row):
    invoice = str(row['Invoice']).strip().upper()
    qty, price = row['Quantity'], row['Price']
    if invoice.startswith('C'):
        return 'CANCELLED_INVOICE'
    if qty < 0:
        return 'RETURN_OR_NEGATIVE_ADJUSTMENT'
    if qty == 0:
        return 'ZERO_QUANTITY'
    if qty > 0:
        if price > 0:  return 'SALE'
        if price == 0: return 'ZERO_PRICE_POSITIVE_QTY'
        if price < 0:  return 'NEGATIVE_PRICE_POSITIVE_QTY'
    return 'OTHER_ADJUSTMENT'

cleaned_transactions['transaction_type'] = cleaned_transactions.apply(
    assign_transaction_type, axis=1)

# ── Boolean flags ─────────────────────────────────────────────────────────────
cleaned_transactions['is_cancelled']         = cleaned_transactions['transaction_type'] == 'CANCELLED_INVOICE'
cleaned_transactions['is_negative_quantity'] = cleaned_transactions['Quantity'] < 0
cleaned_transactions['is_positive_sale']     = cleaned_transactions['transaction_type'] == 'SALE'
cleaned_transactions['revenue']              = cleaned_transactions['Quantity'] * cleaned_transactions['Price']

# ── Canonical description ─────────────────────────────────────────────────────
def modal_description(s):
    nn = s.dropna()
    return nn.mode().iloc[0] if len(nn) > 0 else None

canonical_desc_map = cleaned_transactions.groupby('StockCode')['Description'].agg(modal_description)
cleaned_transactions['canonical_description'] = cleaned_transactions['StockCode'].map(canonical_desc_map)

# ── Analytical datasets ───────────────────────────────────────────────────────
valid_merchandise_transactions = cleaned_transactions[
    (cleaned_transactions['transaction_type'] == 'SALE') &
    (cleaned_transactions['product_type'] == 'MERCHANDISE')
].copy()

customer_transactions = valid_merchandise_transactions[
    valid_merchandise_transactions['Customer ID'].notna()
].copy()

merchandise_product_transactions = cleaned_transactions[
    cleaned_transactions['product_type'] == 'MERCHANDISE'
].copy()

print(f"\nDatasets ready:")
print(f"  cleaned_transactions              : {len(cleaned_transactions):,}")
print(f"  valid_merchandise_transactions    : {len(valid_merchandise_transactions):,}")
print(f"  customer_transactions             : {len(customer_transactions):,}")
print(f"  merchandise_product_transactions  : {len(merchandise_product_transactions):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA TYPE STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — Data type standardization")
print("=" * 70)

for df_name, df in [
    ('cleaned_transactions', cleaned_transactions),
    ('valid_merchandise_transactions', valid_merchandise_transactions),
    ('customer_transactions', customer_transactions),
]:
    # Accept both legacy object and pandas StringDtype (pandas >= 2.0 may infer StringDtype)
    assert df['Invoice'].dtype == object or pd.api.types.is_string_dtype(df['Invoice']), \
        f"{df_name}: Invoice not string-like (dtype={df['Invoice'].dtype})"
    assert df['StockCode'].dtype == object or pd.api.types.is_string_dtype(df['StockCode']), \
        f"{df_name}: StockCode not string-like (dtype={df['StockCode'].dtype})"
    assert pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']), \
        f"{df_name}: InvoiceDate not datetime"
    assert pd.api.types.is_numeric_dtype(df['Quantity']), f"{df_name}: Quantity not numeric"
    assert pd.api.types.is_numeric_dtype(df['Price']),    f"{df_name}: Price not numeric"
    assert pd.api.types.is_numeric_dtype(df['revenue']),  f"{df_name}: revenue not numeric"

print("All dtype assertions passed.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TIME FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — Time features")
print("=" * 70)

MONTH_ORDER   = ['January','February','March','April','May','June',
                 'July','August','September','October','November','December']
DOW_ORDER     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

for df in [cleaned_transactions, valid_merchandise_transactions,
           customer_transactions, merchandise_product_transactions]:
    dt = df['InvoiceDate']
    df['year']             = dt.dt.year
    df['quarter']          = dt.dt.quarter
    df['month']            = dt.dt.month
    df['month_name']       = pd.Categorical(dt.dt.strftime('%B'), categories=MONTH_ORDER, ordered=True)
    df['year_month']       = dt.dt.to_period('M')
    df['week']             = dt.dt.isocalendar().week.astype('Int64')
    df['day']              = dt.dt.day
    df['day_of_week']      = dt.dt.dayofweek          # 0=Mon
    df['day_of_week_name'] = pd.Categorical(dt.dt.strftime('%A'), categories=DOW_ORDER, ordered=True)
    df['hour']             = dt.dt.hour

# Verify chronological sort of year_month
ym_sorted = customer_transactions['year_month'].dropna().sort_values()
assert list(ym_sorted) == sorted(ym_sorted.tolist()), "year_month does not sort chronologically"

print("Time features added and sort-order assertion passed.")
print(f"  Date range: {cleaned_transactions['InvoiceDate'].min()} to {cleaned_transactions['InvoiceDate'].max()}")
print(f"  Years: {sorted(cleaned_transactions['year'].dropna().unique().tolist())}")
print(f"  Months covered: {cleaned_transactions['year_month'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRANSACTION-LEVEL & INVOICE-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — Invoice-level aggregates")
print("=" * 70)

# Invoice-level aggregates on valid_merchandise_transactions
invoice_agg = (
    valid_merchandise_transactions
    .groupby('Invoice')
    .agg(
        invoice_revenue       =('revenue', 'sum'),
        invoice_units         =('Quantity', 'sum'),
        unique_products_per_invoice=('StockCode', 'nunique'),
        invoice_date          =('InvoiceDate', 'first'),
        customer_id           =('Customer ID', 'first'),
        country               =('Country', 'first'),
    )
    .reset_index()
)

print(f"Invoice-level table rows: {len(invoice_agg):,}")
print(f"  mean invoice_revenue : £{invoice_agg['invoice_revenue'].mean():,.2f}")
print(f"  mean invoice_units   : {invoice_agg['invoice_units'].mean():,.1f}")
print(f"  mean unique_products : {invoice_agg['unique_products_per_invoice'].mean():,.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 & 6 — METRIC DEFINITIONS + EXECUTIVE KPI TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5/6 — Executive KPIs")
print("=" * 70)

# --- Gross merchandise sales ---
# Dataset: valid_merchandise_transactions (SALE, MERCHANDISE, Price>0, Qty>0)
gross_revenue = valid_merchandise_transactions['revenue'].sum()

# --- Return/cancellation value (merchandise) ---
# Dataset: merchandise_product_transactions where transaction is negative (cancelled or return)
merch_negative = merchandise_product_transactions[merchandise_product_transactions['revenue'] < 0]
return_value = merch_negative['revenue'].sum()           # negative number
return_value_abs = abs(return_value)                     # absolute for display

# --- Net merchandise revenue ---
# Definition: gross + return_value  (return_value is already negative)
net_revenue = gross_revenue + return_value

# --- Orders ---
# Unique invoices in valid_merchandise_transactions (genuine purchases)
n_orders = valid_merchandise_transactions['Invoice'].nunique()

# --- Units sold ---
units_sold = valid_merchandise_transactions['Quantity'].sum()

# --- Unique customers ---
n_customers = customer_transactions['Customer ID'].nunique()

# --- Unique products ---
n_products = valid_merchandise_transactions['StockCode'].nunique()

# --- Average Order Value (gross, per unique invoice) ---
aov = invoice_agg['invoice_revenue'].mean()

# --- Average items per order ---
avg_items = invoice_agg['invoice_units'].mean()

# --- Repeat customers ---
# Customer with >1 unique invoice in customer_transactions
customer_order_counts = customer_transactions.groupby('Customer ID')['Invoice'].nunique()
repeat_customers   = (customer_order_counts > 1).sum()
total_id_customers = customer_order_counts.shape[0]
repeat_rate        = repeat_customers / total_id_customers * 100

# --- Cancellation invoices ---
cancel_invoices = cleaned_transactions[cleaned_transactions['is_cancelled']]['Invoice'].nunique()
# Rate: cancellation invoices / (cancellation + normal merchandise invoices)
total_invoice_base = cleaned_transactions['Invoice'].nunique()
cancel_rate = cancel_invoices / total_invoice_base * 100

kpis = {
    'Gross Merchandise Revenue (£)':    f'{gross_revenue:,.2f}',
    'Return / Cancellation Value (£)':  f'{return_value_abs:,.2f}',
    'Net Merchandise Revenue (£)':      f'{net_revenue:,.2f}',
    'Orders (unique invoices)':         f'{n_orders:,}',
    'Units Sold':                       f'{units_sold:,}',
    'Unique Customers':                 f'{n_customers:,}',
    'Unique Products':                  f'{n_products:,}',
    'Average Order Value (£)':          f'{aov:,.2f}',
    'Average Items per Order':          f'{avg_items:,.1f}',
    'Repeat Customers':                 f'{repeat_customers:,}',
    'Repeat Customer Rate (%)':         f'{repeat_rate:.2f}%',
    'Cancellation Invoice Rate (%)':    f'{cancel_rate:.2f}%',
}

print("\nEXECUTIVE KPI TABLE")
print("-" * 55)
for k, v in kpis.items():
    print(f"  {k:<40}: {v}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MONTHLY KPI TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7 — Monthly KPI table")
print("=" * 70)

# Gross & net from merchandise_product_transactions (includes cancellations)
monthly_merch = (
    merchandise_product_transactions
    .groupby('year_month')
    .agg(
        gross_revenue =('revenue', lambda x: x[x > 0].sum()),
        return_value  =('revenue', lambda x: x[x < 0].sum()),
    )
    .reset_index()
)
monthly_merch['net_revenue'] = monthly_merch['gross_revenue'] + monthly_merch['return_value']
monthly_merch['return_value_abs'] = monthly_merch['return_value'].abs()

# Orders, units, customers from valid_merchandise_transactions
monthly_sales = (
    valid_merchandise_transactions
    .groupby('year_month')
    .agg(
        orders           =('Invoice', 'nunique'),
        units            =('Quantity', 'sum'),
    )
    .reset_index()
)

# Customer count from customer_transactions (only identified)
monthly_cust = (
    customer_transactions
    .groupby('year_month')['Customer ID']
    .nunique()
    .reset_index()
    .rename(columns={'Customer ID': 'unique_customers'})
)

# Average order value — from invoice_agg
invoice_agg['year_month'] = invoice_agg['invoice_date'].dt.to_period('M')
monthly_aov = (
    invoice_agg
    .groupby('year_month')['invoice_revenue']
    .mean()
    .reset_index()
    .rename(columns={'invoice_revenue': 'average_order_value'})
)

# Merge all monthly tables
monthly_kpis = (
    monthly_merch
    .merge(monthly_sales, on='year_month', how='outer')
    .merge(monthly_cust,  on='year_month', how='outer')
    .merge(monthly_aov,   on='year_month', how='outer')
    .sort_values('year_month')
    .reset_index(drop=True)
)

# Monthly growth in net_revenue
monthly_kpis['net_revenue_growth_pct'] = monthly_kpis['net_revenue'].pct_change() * 100
# Mask growth where previous period was zero (undefined) or is the first period
monthly_kpis.loc[monthly_kpis['net_revenue'].shift(1) == 0, 'net_revenue_growth_pct'] = np.nan

print(monthly_kpis[['year_month','gross_revenue','return_value_abs','net_revenue',
                     'orders','units','unique_customers','average_order_value',
                     'net_revenue_growth_pct']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — YEARLY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — Yearly KPI table")
print("=" * 70)

yearly_merch = (
    merchandise_product_transactions
    .groupby('year')
    .agg(
        gross_revenue=('revenue', lambda x: x[x > 0].sum()),
        return_value =('revenue', lambda x: x[x < 0].sum()),
    )
    .reset_index()
)
yearly_merch['net_revenue'] = yearly_merch['gross_revenue'] + yearly_merch['return_value']
yearly_merch['return_value_abs'] = yearly_merch['return_value'].abs()

yearly_sales = (
    valid_merchandise_transactions
    .groupby('year')
    .agg(orders=('Invoice','nunique'), units=('Quantity','sum'))
    .reset_index()
)

yearly_cust = (
    customer_transactions
    .groupby('year')['Customer ID']
    .nunique()
    .reset_index()
    .rename(columns={'Customer ID':'unique_customers'})
)

invoice_agg['year'] = invoice_agg['invoice_date'].dt.year
yearly_aov = (
    invoice_agg
    .groupby('year')['invoice_revenue']
    .mean()
    .reset_index()
    .rename(columns={'invoice_revenue':'average_order_value'})
)

yearly_kpis = (
    yearly_merch
    .merge(yearly_sales, on='year', how='outer')
    .merge(yearly_cust,  on='year', how='outer')
    .merge(yearly_aov,   on='year', how='outer')
    .sort_values('year')
    .reset_index(drop=True)
)

# YoY change
for col in ['gross_revenue','net_revenue','orders','units']:
    yearly_kpis[f'{col}_yoy_pct'] = yearly_kpis[col].pct_change() * 100

print(yearly_kpis.to_string(index=False))

# Report number of months per year
months_per_year = cleaned_transactions.groupby('year')['year_month'].nunique()
print("\nMonths represented per year (partial-year caveat):")
for yr, nm in months_per_year.items():
    print(f"  {yr}: {nm} months {'(PARTIAL)' if nm < 12 else ''}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CUSTOMER SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9 — Customer summary table")
print("=" * 70)

customer_summary = (
    customer_transactions
    .groupby('Customer ID')
    .agg(
        first_purchase_date =('InvoiceDate', 'min'),
        last_purchase_date  =('InvoiceDate', 'max'),
        total_orders        =('Invoice', 'nunique'),
        total_units         =('Quantity', 'sum'),
        total_revenue       =('revenue', 'sum'),
        unique_products     =('StockCode', 'nunique'),
    )
    .reset_index()
    .rename(columns={'Customer ID': 'customer_id'})
)

customer_summary['average_order_value'] = (
    customer_summary['total_revenue'] / customer_summary['total_orders']
)
customer_summary['customer_lifetime_days'] = (
    (customer_summary['last_purchase_date'] - customer_summary['first_purchase_date']).dt.days
)

print(f"Customers in summary: {len(customer_summary):,}")
print(customer_summary.describe().to_string())

# Reconciliation: customer_summary revenue vs customer_transactions revenue
cust_summary_revenue = customer_summary['total_revenue'].sum()
cust_txn_revenue     = customer_transactions['revenue'].sum()
discrepancy          = abs(cust_summary_revenue - cust_txn_revenue)
print(f"\nReconciliation:")
print(f"  customer_summary.total_revenue.sum() : £{cust_summary_revenue:,.2f}")
print(f"  customer_transactions.revenue.sum()  : £{cust_txn_revenue:,.2f}")
print(f"  Discrepancy                          : £{discrepancy:,.6f}")
assert discrepancy < 0.01, f"Customer revenue reconciliation failed: £{discrepancy:.4f}"
print("  [PASS] Customer revenue reconciles.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PRODUCT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 10 — Product summary table")
print("=" * 70)

# Gross sales (positive only) from valid_merchandise_transactions
product_gross = (
    valid_merchandise_transactions
    .groupby('StockCode')
    .agg(
        canonical_description=('canonical_description', 'first'),
        total_units_sold     =('Quantity', 'sum'),
        gross_revenue        =('revenue', 'sum'),
        number_of_orders     =('Invoice', 'nunique'),
        number_of_customers  =('Customer ID', 'nunique'),
    )
    .reset_index()
)

# Return/cancellation value (negative rows) from merchandise_product_transactions
product_returns = (
    merchandise_product_transactions[merchandise_product_transactions['revenue'] < 0]
    .groupby('StockCode')['revenue']
    .sum()
    .reset_index()
    .rename(columns={'revenue': 'return_value'})
)

# Use outer join: some StockCodes may have returns but no positive sales rows
product_summary = product_gross.merge(product_returns, on='StockCode', how='outer')
product_summary['return_value']  = product_summary['return_value'].fillna(0)
product_summary['gross_revenue'] = product_summary['gross_revenue'].fillna(0)
product_summary['total_units_sold'] = product_summary['total_units_sold'].fillna(0)
product_summary['number_of_orders'] = product_summary['number_of_orders'].fillna(0)
product_summary['number_of_customers'] = product_summary['number_of_customers'].fillna(0)
# Fill canonical_description for return-only products
product_summary['canonical_description'] = product_summary['canonical_description'].fillna(
    product_summary['StockCode'].map(canonical_desc_map)
)
product_summary['net_revenue'] = product_summary['gross_revenue'] + product_summary['return_value']

print(f"Products in summary: {len(product_summary):,}")

# Reconciliation
prod_gross_total  = product_summary['gross_revenue'].sum()
vmt_gross_total   = valid_merchandise_transactions['revenue'].sum()
prod_net_total    = product_summary['net_revenue'].sum()
mpt_net_total     = merchandise_product_transactions['revenue'].sum()

discrepancy_gross = abs(prod_gross_total - vmt_gross_total)
discrepancy_net   = abs(prod_net_total - mpt_net_total)
print(f"\nProduct-level reconciliation:")
print(f"  product_summary gross   : {prod_gross_total:,.2f}")
print(f"  vmt gross               : {vmt_gross_total:,.2f}")
print(f"  discrepancy (gross)     : {discrepancy_gross:,.4f}")
print(f"  product_summary net     : {prod_net_total:,.2f}")
print(f"  mpt net                 : {mpt_net_total:,.2f}")
print(f"  discrepancy (net)       : {discrepancy_net:,.4f}")
assert discrepancy_gross < 0.01, f"Product gross reconciliation failed: {discrepancy_gross:.4f}"
assert discrepancy_net   < 0.01, f"Product net reconciliation failed: {discrepancy_net:.4f}"
print("  [PASS] Product revenue reconciles.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — GEOGRAPHIC SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 11 — Geographic summary table")
print("=" * 70)

# Transaction metrics from valid_merchandise_transactions (all rows, not just identified)
country_gross = (
    valid_merchandise_transactions
    .groupby('Country')
    .agg(
        gross_revenue=('revenue', 'sum'),
        orders       =('Invoice', 'nunique'),
        units_sold   =('Quantity', 'sum'),
    )
    .reset_index()
)

# Customer-level metrics from customer_transactions
country_cust = (
    customer_transactions
    .groupby('Country')
    .agg(
        unique_customers=('Customer ID', 'nunique'),
    )
    .reset_index()
)

# Return value from merchandise_product_transactions
country_returns = (
    merchandise_product_transactions[merchandise_product_transactions['revenue'] < 0]
    .groupby('Country')['revenue']
    .sum()
    .reset_index()
    .rename(columns={'revenue': 'return_value'})
)

country_summary = (
    country_gross
    .merge(country_cust,   on='Country', how='left')
    .merge(country_returns, on='Country', how='left')
)
country_summary['return_value']   = country_summary['return_value'].fillna(0)
country_summary['net_revenue']    = country_summary['gross_revenue'] + country_summary['return_value']
country_summary['unique_customers'] = country_summary['unique_customers'].fillna(0).astype(int)
country_summary['average_order_value'] = country_summary['gross_revenue'] / country_summary['orders']
country_summary['revenue_per_customer'] = np.where(
    country_summary['unique_customers'] > 0,
    country_summary['gross_revenue'] / country_summary['unique_customers'],
    np.nan
)
country_summary = country_summary.sort_values('gross_revenue', ascending=False).reset_index(drop=True)

print(f"Countries: {len(country_summary)}")
print(country_summary.head(15).to_string(index=False))

# Reconciliation: country totals vs valid_merchandise_transactions
country_gross_total = country_summary['gross_revenue'].sum()
vmt_gross_check     = valid_merchandise_transactions['revenue'].sum()
discrepancy_country = abs(country_gross_total - vmt_gross_check)
assert discrepancy_country < 0.01, f"Country reconciliation failed: £{discrepancy_country:.4f}"
print(f"\n  [PASS] Country gross revenue reconciles (discrepancy £{discrepancy_country:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 12 — Generating visualizations")
print("=" * 70)

def save_fig(fname):
    p = os.path.join(FIGURES_DIR, fname)
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p}")

fmt_k  = mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
fmt_gbp = mticker.FuncFormatter(lambda x, _: f'£{x/1e6:.1f}M' if abs(x) >= 1e6 else f'£{x:,.0f}')

# ── Chart 1: KPI summary block ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
kpi_display = [
    ('Gross Revenue',      f"£{gross_revenue:,.0f}"),
    ('Return Value',       f"£{return_value_abs:,.0f}"),
    ('Net Revenue',        f"£{net_revenue:,.0f}"),
    ('Orders',             f"{n_orders:,}"),
    ('Units Sold',         f"{units_sold:,}"),
    ('Unique Customers',   f"{n_customers:,}"),
    ('Unique Products',    f"{n_products:,}"),
    ('Avg Order Value',    f"£{aov:,.2f}"),
    ('Repeat Rate',        f"{repeat_rate:.1f}%"),
    ('Cancel Rate',        f"{cancel_rate:.1f}%"),
]
n_kpi = len(kpi_display)
cols = 5
rows_k = (n_kpi + cols - 1) // cols
for i, (label, val) in enumerate(kpi_display):
    col_i = i % cols
    row_i = i // cols
    x = col_i / cols + 0.02
    y = 1.0 - row_i / rows_k - 0.15
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, color='#57606a', va='top')
    ax.text(x, y - 0.18, val, transform=ax.transAxes,
            fontsize=15, fontweight='bold', color='#1f2328', va='top')
ax.set_title('Executive KPI Summary — All Periods', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
save_fig('09_kpi_summary.png')

# ── Chart 2: Monthly gross vs net revenue ─────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ym_labels = [str(p) for p in monthly_kpis['year_month']]
x = range(len(ym_labels))
ax.bar(x, monthly_kpis['gross_revenue'], label='Gross Revenue', alpha=0.85,
       color=sns.color_palette('muted')[0], edgecolor='white')
ax.bar(x, monthly_kpis['net_revenue'], label='Net Revenue', alpha=0.85,
       color=sns.color_palette('muted')[2], edgecolor='white')
ax.set_xticks(list(x))
ax.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(fmt_gbp)
ax.set_title('Monthly Gross vs Net Merchandise Revenue', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Revenue (£)')
ax.legend()
plt.tight_layout()
save_fig('10_monthly_gross_vs_net_revenue.png')

# ── Chart 3: Monthly order count ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(list(x), monthly_kpis['orders'].fillna(0),
       color=sns.color_palette('muted')[1], edgecolor='white')
ax.set_xticks(list(x))
ax.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(fmt_k)
ax.set_title('Monthly Order Count (Unique Merchandise Invoices)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Orders')
plt.tight_layout()
save_fig('11_monthly_orders.png')

# ── Chart 4: Monthly unique customers ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(list(x), monthly_kpis['unique_customers'].fillna(0),
       color=sns.color_palette('muted')[3], edgecolor='white')
ax.set_xticks(list(x))
ax.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(fmt_k)
ax.set_title('Monthly Unique Customers (Identified)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Unique Customers')
plt.tight_layout()
save_fig('12_monthly_unique_customers.png')

# ── Chart 5: Monthly AOV ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(list(x), monthly_kpis['average_order_value'].fillna(np.nan),
        marker='o', linewidth=2, color=sns.color_palette('muted')[4], markersize=4)
ax.set_xticks(list(x))
ax.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'£{v:,.0f}'))
ax.set_title('Monthly Average Order Value (Gross Merchandise)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('AOV (£)')
plt.tight_layout()
save_fig('13_monthly_aov.png')

# ── Chart 6: Yearly net revenue ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
yr_labels = [str(y) for y in yearly_kpis['year']]
bars = ax.bar(yr_labels, yearly_kpis['net_revenue'],
              color=sns.color_palette('muted', n_colors=len(yr_labels)), edgecolor='white')
for bar, val in zip(bars, yearly_kpis['net_revenue']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + yearly_kpis['net_revenue'].max() * 0.01,
            f'£{val/1e6:.2f}M', ha='center', va='bottom', fontsize=10)
ax.yaxis.set_major_formatter(fmt_gbp)
ax.set_title('Yearly Net Merchandise Revenue\n(partial years noted)', fontsize=12, fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Net Revenue (£)')
# Annotate partial years
for yr, nm in months_per_year.items():
    if nm < 12:
        idx = yr_labels.index(str(yr))
        ax.annotate(f'({nm} months)', xy=(idx, 0),
                    xytext=(idx, -yearly_kpis['net_revenue'].max() * 0.08),
                    ha='center', fontsize=8, color='#57606a')
plt.tight_layout()
save_fig('14_yearly_net_revenue.png')

# ── Chart 7: Customer revenue distribution ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
p99 = customer_summary['total_revenue'].quantile(0.99)
clipped = customer_summary['total_revenue'][customer_summary['total_revenue'] <= p99]
ax.hist(clipped, bins=60, color=sns.color_palette('muted')[0], edgecolor='white', linewidth=0.4)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'£{v:,.0f}'))
ax.yaxis.set_major_formatter(fmt_k)
ax.set_title(f'Customer Total Revenue Distribution (capped at 99th pct = £{p99:,.0f})',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Total Revenue per Customer (£)'); ax.set_ylabel('Customers')
plt.tight_layout()
save_fig('15_customer_revenue_distribution.png')

# ── Chart 8: Order value distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
p99_inv = invoice_agg['invoice_revenue'].quantile(0.99)
clipped_inv = invoice_agg['invoice_revenue'][invoice_agg['invoice_revenue'] <= p99_inv]
ax.hist(clipped_inv, bins=60, color=sns.color_palette('muted')[1], edgecolor='white', linewidth=0.4)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'£{v:,.0f}'))
ax.yaxis.set_major_formatter(fmt_k)
ax.set_title(f'Order Value Distribution (capped at 99th pct = £{p99_inv:,.0f})',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Order Value (£)'); ax.set_ylabel('Orders')
plt.tight_layout()
save_fig('16_order_value_distribution.png')

# ── Chart 9: Top 10 countries by net revenue ─────────────────────────────
top10_countries = country_summary.head(10)
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top10_countries['Country'][::-1],
        top10_countries['net_revenue'][::-1],
        color=sns.color_palette('muted', n_colors=10)[::-1], edgecolor='white')
ax.xaxis.set_major_formatter(fmt_gbp)
ax.set_title('Top 10 Countries by Net Merchandise Revenue', fontsize=13, fontweight='bold')
ax.set_xlabel('Net Revenue (£)'); ax.set_ylabel('Country')
plt.tight_layout()
save_fig('17_top10_countries_net_revenue.png')

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — FULL RECONCILIATION SUITE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 13 — Reconciliation & Validation")
print("=" * 70)

all_pass = True
def chk(label, condition, detail=''):
    global all_pass
    status = 'PASS' if condition else 'FAIL'
    if not condition:
        all_pass = False
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ''))
    return condition

# 1. Revenue = Quantity * Price
rev_check = (cleaned_transactions['Quantity'] * cleaned_transactions['Price']).round(8)
chk("revenue == Quantity * Price (all rows)",
    (rev_check == cleaned_transactions['revenue'].round(8)).all())

# 2. Gross + negative merch = net
computed_net = gross_revenue + return_value
chk("gross + return_value == net_revenue",
    abs(computed_net - net_revenue) < 0.01,
    f"£{computed_net:,.2f} vs £{net_revenue:,.2f}")

# 3. Order count is unique invoices, not rows
rows_in_vmt = len(valid_merchandise_transactions)
chk("Orders (invoices) < rows in valid_merchandise_transactions",
    n_orders < rows_in_vmt,
    f"{n_orders:,} orders vs {rows_in_vmt:,} rows")

# 4. Monthly totals reconcile to overall
monthly_net_sum = monthly_kpis['net_revenue'].sum()
chk("Monthly net_revenue sums to overall net",
    abs(monthly_net_sum - net_revenue) < 1.0,
    f"monthly sum £{monthly_net_sum:,.2f} vs overall £{net_revenue:,.2f}")

monthly_order_sum = monthly_kpis['orders'].sum()
chk("Monthly orders sum to total orders",
    abs(monthly_order_sum - n_orders) < 1,
    f"monthly sum {int(monthly_order_sum):,} vs total {n_orders:,}")

# 5. Yearly totals reconcile
yearly_net_sum = yearly_kpis['net_revenue'].sum()
chk("Yearly net_revenue sums to overall net",
    abs(yearly_net_sum - net_revenue) < 1.0,
    f"yearly sum £{yearly_net_sum:,.2f} vs overall £{net_revenue:,.2f}")

# 6. Customer-level revenue reconciles
chk("Customer-level revenue reconciles",
    abs(customer_summary['total_revenue'].sum() - customer_transactions['revenue'].sum()) < 0.01)

# 7. Product-level gross revenue reconciles
chk("Product-level gross revenue reconciles",
    abs(product_summary['gross_revenue'].sum() - valid_merchandise_transactions['revenue'].sum()) < 0.01)

# 8. Country-level gross revenue reconciles
chk("Country-level gross revenue reconciles",
    abs(country_summary['gross_revenue'].sum() - valid_merchandise_transactions['revenue'].sum()) < 0.01)

if all_pass:
    print("\n  All reconciliation checks passed.")
else:
    print("\n  ONE OR MORE CHECKS FAILED — investigate before proceeding.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — EXECUTIVE KPI FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 14 — Executive KPI Findings")
print("=" * 70)

return_rate_pct      = return_value_abs / gross_revenue * 100
uk_row = country_summary[country_summary['Country'] == 'United Kingdom']
uk_net  = uk_row['net_revenue'].values[0] if len(uk_row) > 0 else 0
uk_pct  = uk_net / net_revenue * 100 if net_revenue > 0 else 0
max_month_row  = monthly_kpis.loc[monthly_kpis['net_revenue'].idxmax()]
min_month_row  = monthly_kpis.loc[monthly_kpis['net_revenue'].idxmin()]
one_time_custs = (customer_order_counts == 1).sum()
one_time_pct   = one_time_custs / total_id_customers * 100

findings = [
    {
        'id': 1,
        'finding': f"Gross merchandise revenue across the full observation period is "
                   f"£{gross_revenue:,.0f}, with return/cancellation value of "
                   f"£{return_value_abs:,.0f} ({return_rate_pct:.1f}% of gross), "
                   f"yielding net merchandise revenue of £{net_revenue:,.0f}.",
        'evidence': "gross_revenue and return_value computed from merchandise_product_transactions.",
        'business_relevance': "The gap between gross and net revenue quantifies the economic impact of returns "
                              "and should be tracked as a separate KPI from gross sales."
    },
    {
        'id': 2,
        'finding': f"The dataset contains {n_orders:,} unique merchandise purchase invoices "
                   f"across {n_customers:,} identified customers, with an average order value "
                   f"of £{aov:,.2f} and {avg_items:.1f} items per order.",
        'evidence': "invoice_agg on valid_merchandise_transactions.",
        'business_relevance': "AOV and basket size are baseline metrics for revenue-per-visit and "
                              "upsell/cross-sell effectiveness assessments."
    },
    {
        'id': 3,
        'finding': f"{repeat_customers:,} of {total_id_customers:,} identified customers "
                   f"({repeat_rate:.1f}%) made more than one purchase invoice. "
                   f"{one_time_custs:,} customers ({one_time_pct:.1f}%) made exactly one purchase.",
        'evidence': "customer_order_counts from customer_transactions.",
        'business_relevance': "The repeat customer rate indicates customer loyalty baseline. "
                              "The one-time customer proportion signals retention opportunity."
    },
    {
        'id': 4,
        'finding': f"Peak net merchandise revenue month was {str(max_month_row['year_month'])} "
                   f"(£{max_month_row['net_revenue']:,.0f}). "
                   f"Lowest was {str(min_month_row['year_month'])} "
                   f"(£{min_month_row['net_revenue']:,.0f}).",
        'evidence': "monthly_kpis.net_revenue.idxmax() / idxmin().",
        'business_relevance': "Peak and trough months inform inventory planning, staffing, "
                              "and promotional calendar decisions."
    },
    {
        'id': 5,
        'finding': f"United Kingdom accounts for £{uk_net:,.0f} net merchandise revenue "
                   f"({uk_pct:.1f}% of total net revenue), "
                   f"with {len(country_summary)} countries represented in total.",
        'evidence': "country_summary filtered to United Kingdom.",
        'business_relevance': "High UK concentration means international revenue diversification "
                              "represents a potential growth lever, but also that UK-specific disruptions "
                              "disproportionately affect total revenue."
    },
]

for f in findings:
    print(f"\nFinding {f['id']}:")
    print(f"  Finding           : {f['finding']}")
    print(f"  Evidence          : {f['evidence']}")
    print(f"  Business relevance: {f['business_relevance']}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — SAVE ANALYTICAL TABLES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 15 — Saving analytical tables")
print("=" * 70)

# Save as CSV (lightweight, no raw transaction data)
customer_summary.to_csv(f'{OUTPUTS_DIR}customer_summary.csv', index=False)
product_summary.to_csv(f'{OUTPUTS_DIR}product_summary.csv', index=False)
country_summary.to_csv(f'{OUTPUTS_DIR}country_summary.csv', index=False)

# Convert Period columns to string for CSV compatibility
monthly_kpis_save = monthly_kpis.copy()
monthly_kpis_save['year_month'] = monthly_kpis_save['year_month'].astype(str)
monthly_kpis_save.to_csv(f'{OUTPUTS_DIR}monthly_kpis.csv', index=False)
yearly_kpis.to_csv(f'{OUTPUTS_DIR}yearly_kpis.csv', index=False)

saved_files = [
    'outputs/customer_summary.csv',
    'outputs/product_summary.csv',
    'outputs/country_summary.csv',
    'outputs/monthly_kpis.csv',
    'outputs/yearly_kpis.csv',
]
for f in saved_files:
    print(f"  Saved: {f}")

print("\n" + "=" * 70)
print("EXECUTION COMPLETE — all sections ran successfully.")
print("=" * 70)

# ── Print final KPI summary for reporting ────────────────────────────────────
print("\n=== FINAL KPI VALUES (for notebook embedding) ===")
print(f"  RAW_ROW_COUNT                    : {RAW_ROW_COUNT}")
print(f"  n_duplicates                     : {n_duplicates}")
print(f"  len(cleaned_transactions)        : {len(cleaned_transactions)}")
print(f"  len(valid_merchandise_txns)      : {len(valid_merchandise_transactions)}")
print(f"  len(customer_transactions)       : {len(customer_transactions)}")
print(f"  len(merchandise_product_txns)    : {len(merchandise_product_transactions)}")
print(f"  gross_revenue                    : {gross_revenue}")
print(f"  return_value_abs                 : {return_value_abs}")
print(f"  net_revenue                      : {net_revenue}")
print(f"  n_orders                         : {n_orders}")
print(f"  units_sold                       : {units_sold}")
print(f"  n_customers                      : {n_customers}")
print(f"  n_products                       : {n_products}")
print(f"  aov                              : {aov}")
print(f"  avg_items                        : {avg_items}")
print(f"  repeat_customers                 : {repeat_customers}")
print(f"  total_id_customers               : {total_id_customers}")
print(f"  repeat_rate                      : {repeat_rate}")
print(f"  cancel_rate                      : {cancel_rate}")
print(f"  one_time_custs                   : {one_time_custs}")
print(f"  one_time_pct                     : {one_time_pct}")
print(f"  uk_net                           : {uk_net}")
print(f"  uk_pct                           : {uk_pct}")
print(f"  n_countries                      : {len(country_summary)}")
print(f"  max_month                        : {str(max_month_row['year_month'])}")
print(f"  max_month_net                    : {max_month_row['net_revenue']}")
print(f"  min_month                        : {str(min_month_row['year_month'])}")
print(f"  min_month_net                    : {min_month_row['net_revenue']}")
# Description stats derived on the fly from cleaned_transactions
desc_per_stock_final = (
    cleaned_transactions.dropna(subset=['Description'])
    .groupby('StockCode')['Description'].nunique()
)
n_single_desc = (desc_per_stock_final == 1).sum()
n_multi_desc  = (desc_per_stock_final > 1).sum()
print(f"  n_single_desc                    : {n_single_desc}")
print(f"  n_multi_desc                     : {n_multi_desc}")
print(f"  months_per_year                  : {dict(months_per_year)}")
print(f"  len(monthly_kpis)                : {len(monthly_kpis)}")
print(f"  len(yearly_kpis)                 : {len(yearly_kpis)}")
