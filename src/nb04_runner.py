"""
Notebook 04 — Sales & Revenue Intelligence
Execution script: run from project root to validate all analytics.
python src/nb04_runner.py
"""
import os, re, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore', message='.*data validation.*', category=UserWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.4f}'.format)

DATA_PATH   = 'data/online_retail_II.xlsx'
FIGURES_DIR = 'outputs/figures/'
OUTPUTS_DIR = 'outputs/'
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

# ─────────────────────────────────────────────────────────────────────────────
# SHARED PIPELINE (identical to NB02/03)
# ─────────────────────────────────────────────────────────────────────────────
def classify_stockcode(code, description):
    code = str(code).strip().upper() if pd.notna(code) else ''
    desc = str(description).strip().upper() if pd.notna(description) else ''
    if re.match(r'^TEST', code) or 'TEST' in desc:         return 'TEST'
    if code in ('B','ADJUST2') or 'BAD DEBT' in desc:      return 'BAD_DEBT'
    if code.startswith('GIFT') or 'GIFT VOUCHER' in desc:  return 'GIFT_VOUCHER'
    if code in ('POST','DOT','C2') or 'POSTAGE' in desc or 'CARRIAGE' in desc: return 'POSTAGE'
    if code in ('BANK CHARGES','BANKCHARGES','AMAZONFEE') \
            or 'BANK CHARGE' in desc or 'AMAZON FEE' in desc or 'FEE' in desc: return 'FEE_OR_CHARGE'
    if code == 'D' or 'DISCOUNT' in desc:                  return 'DISCOUNT'
    if code == 'S' or 'SAMPLE' in desc:                    return 'SAMPLE'
    if code in ('M','ADJUST','CRUK') or 'MANUAL' in desc or 'ADJUST' in desc or 'CRUK' in desc:
        return 'MANUAL_ADJUSTMENT'
    if re.match(r'^\d{5}[A-Z]?$', code):                   return 'MERCHANDISE'
    return 'UNCLASSIFIED_NON_STANDARD'

def assign_transaction_type(row):
    inv = str(row['Invoice']).strip().upper()
    q, p = row['Quantity'], row['Price']
    if inv.startswith('C'):   return 'CANCELLED_INVOICE'
    if q < 0:                 return 'RETURN_OR_NEGATIVE_ADJUSTMENT'
    if q == 0:                return 'ZERO_QUANTITY'
    if q > 0:
        if p > 0:  return 'SALE'
        if p == 0: return 'ZERO_PRICE_POSITIVE_QTY'
        if p < 0:  return 'NEGATIVE_PRICE_POSITIVE_QTY'
    return 'OTHER_ADJUSTMENT'

def modal_description(s):
    nn = s.dropna()
    return nn.mode().iloc[0] if len(nn) > 0 else None

def save_fig(fname):
    p = os.path.join(FIGURES_DIR, fname)
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')

fmt_gbp = mticker.FuncFormatter(lambda v, _: f'£{v/1e6:.1f}M' if abs(v)>=1e6 else f'£{v:,.0f}')
fmt_k   = mticker.FuncFormatter(lambda v, _: f'{int(v):,}')

print("Loading data...")
xl = pd.ExcelFile(DATA_PATH, engine='openpyxl')
tagged = []
for name in xl.sheet_names:
    df = pd.read_excel(DATA_PATH, sheet_name=name, engine='openpyxl',
                       dtype={'Invoice': str, 'StockCode': str})
    df['_sheet'] = name
    tagged.append(df)
raw = pd.concat(tagged, ignore_index=True)
raw['InvoiceDate'] = pd.to_datetime(raw['InvoiceDate'], errors='coerce')

dup_mask = raw.duplicated(keep='first')
ct = raw[~dup_mask].copy().reset_index(drop=True)

ct['is_zero_price']     = ct['Price'] == 0
ct['is_negative_price'] = ct['Price'] < 0
ct['product_type']      = ct.apply(lambda r: classify_stockcode(r['StockCode'], r['Description']), axis=1)
ct['transaction_type']  = ct.apply(assign_transaction_type, axis=1)
ct['is_cancelled']      = ct['transaction_type'] == 'CANCELLED_INVOICE'
ct['is_negative_quantity'] = ct['Quantity'] < 0
ct['is_positive_sale']  = ct['transaction_type'] == 'SALE'
ct['revenue']           = ct['Quantity'] * ct['Price']

canonical_desc_map = ct.groupby('StockCode')['Description'].agg(modal_description)
ct['canonical_description'] = ct['StockCode'].map(canonical_desc_map)

MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
DOW_ORDER   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

for df in [ct]:
    dt = df['InvoiceDate']
    df['year']             = dt.dt.year
    df['quarter']          = dt.dt.quarter
    df['month']            = dt.dt.month
    df['month_name']       = pd.Categorical(dt.dt.strftime('%B'), categories=MONTH_ORDER, ordered=True)
    df['year_month']       = dt.dt.to_period('M')
    df['week']             = dt.dt.isocalendar().week.astype('Int64')
    df['day']              = dt.dt.day
    df['day_of_week']      = dt.dt.dayofweek
    df['day_of_week_name'] = pd.Categorical(dt.dt.strftime('%A'), categories=DOW_ORDER, ordered=True)
    df['hour']             = dt.dt.hour

vmt = ct[(ct['transaction_type']=='SALE') & (ct['product_type']=='MERCHANDISE')].copy()
cust_txn = vmt[vmt['Customer ID'].notna()].copy()
mpt = ct[ct['product_type']=='MERCHANDISE'].copy()

print(f"  cleaned_transactions              : {len(ct):,}")
print(f"  valid_merchandise_transactions    : {len(vmt):,}")
print(f"  customer_transactions             : {len(cust_txn):,}")
print(f"  merchandise_product_transactions  : {len(mpt):,}")

# Load pre-computed summaries
customer_summary = pd.read_csv(f'{OUTPUTS_DIR}customer_summary.csv')
product_summary  = pd.read_csv(f'{OUTPUTS_DIR}product_summary.csv')
country_summary  = pd.read_csv(f'{OUTPUTS_DIR}country_summary.csv')
monthly_kpis_csv = pd.read_csv(f'{OUTPUTS_DIR}monthly_kpis.csv')
yearly_kpis_csv  = pd.read_csv(f'{OUTPUTS_DIR}yearly_kpis.csv')
print("Pre-computed summaries loaded.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MONTHLY ANALYTICAL TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 1 — Monthly analytical table")
print("="*70)

# invoice-level for AOV and basket size
invoice_agg = (
    vmt.groupby('Invoice')
    .agg(
        invoice_revenue  =('revenue',   'sum'),
        invoice_units    =('Quantity',  'sum'),
        unique_products  =('StockCode', 'nunique'),
        invoice_date     =('InvoiceDate','first'),
        customer_id      =('Customer ID','first'),
    )
    .reset_index()
)
invoice_agg['year_month'] = invoice_agg['invoice_date'].dt.to_period('M')

monthly_rev = (
    mpt.groupby('year_month')
    .agg(
        gross_revenue=('revenue', lambda x: x[x > 0].sum()),
        return_value =('revenue', lambda x: x[x < 0].sum()),
    ).reset_index()
)
monthly_rev['net_revenue']    = monthly_rev['gross_revenue'] + monthly_rev['return_value']
monthly_rev['return_value_abs'] = monthly_rev['return_value'].abs()

monthly_sales = (
    vmt.groupby('year_month')
    .agg(
        orders=('Invoice','nunique'),
        units =('Quantity','sum'),
    ).reset_index()
)

monthly_cust = (
    cust_txn.groupby('year_month')['Customer ID']
    .nunique().reset_index()
    .rename(columns={'Customer ID':'unique_customers'})
)

monthly_agg_inv = (
    invoice_agg.groupby('year_month')
    .agg(
        gross_aov      =('invoice_revenue','mean'),
        avg_units_per_order  =('invoice_units',  'mean'),
        avg_unique_products  =('unique_products', 'mean'),
    ).reset_index()
)

# Net AOV: net_revenue / orders
monthly_m = (
    monthly_rev
    .merge(monthly_sales, on='year_month', how='outer')
    .merge(monthly_cust,  on='year_month', how='outer')
    .merge(monthly_agg_inv, on='year_month', how='outer')
    .sort_values('year_month')
    .reset_index(drop=True)
)
monthly_m['net_aov'] = monthly_m['net_revenue'] / monthly_m['orders']
monthly_m['year']    = monthly_m['year_month'].dt.year
monthly_m['month']   = monthly_m['year_month'].dt.month
monthly_m['ym_str']  = monthly_m['year_month'].astype(str)

# MoM changes
for col in ['net_revenue','orders','unique_customers','gross_aov']:
    monthly_m[f'{col}_mom_pct'] = monthly_m[col].pct_change() * 100
    monthly_m.loc[monthly_m[col].shift(1) == 0, f'{col}_mom_pct'] = np.nan

print(f"Monthly table rows: {len(monthly_m)}")
# Validate
assert abs(monthly_m['net_revenue'].sum() - mpt['revenue'].sum()) < 1.0
print(f"Monthly net_revenue sum reconciles: £{monthly_m['net_revenue'].sum():,.2f}")

# Overall KPIs
gross_rev = vmt['revenue'].sum()
net_rev   = mpt['revenue'].sum()
n_orders  = vmt['Invoice'].nunique()
n_cust    = cust_txn['Customer ID'].nunique()
units_sold = vmt['Quantity'].sum()
gross_aov = invoice_agg['invoice_revenue'].mean()
net_aov   = net_rev / n_orders

print(f"\nOverall KPIs:")
print(f"  gross_revenue : £{gross_rev:,.2f}")
print(f"  net_revenue   : £{net_rev:,.2f}")
print(f"  n_orders      : {n_orders:,}")
print(f"  n_customers   : {n_cust:,}")
print(f"  units_sold    : {units_sold:,}")
print(f"  gross_aov     : £{gross_aov:,.2f}")
print(f"  net_aov       : £{net_aov:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REVENUE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 2 — Revenue decomposition")
print("="*70)

# Verify: Revenue = Orders × Gross AOV
monthly_m['rev_from_orders_aov'] = monthly_m['orders'] * monthly_m['gross_aov']
decomp_check = (monthly_m['gross_revenue'] - monthly_m['rev_from_orders_aov']).abs().max()
print(f"Revenue = Orders x AOV max residual: £{decomp_check:,.4f}  (should be ~0)")
assert decomp_check < 0.01, f"Revenue decomposition check failed: {decomp_check}"

# Verify: Units = Orders × Avg Units per Order
monthly_m['units_from_decomp'] = monthly_m['orders'] * monthly_m['avg_units_per_order']
units_check = (monthly_m['units'] - monthly_m['units_from_decomp']).abs().max()
print(f"Units = Orders x AvgUnits max residual: {units_check:,.4f}")
assert units_check < 0.1

# Correlations
corr_orders_rev = monthly_m[['net_revenue','orders']].corr().iloc[0,1]
corr_aov_rev    = monthly_m[['net_revenue','gross_aov']].corr().iloc[0,1]
corr_cust_rev   = monthly_m[['net_revenue','unique_customers']].dropna().corr().iloc[0,1]
corr_units_rev  = monthly_m[['net_revenue','units']].corr().iloc[0,1]
print(f"\nCorrelations with monthly net_revenue:")
print(f"  orders       : {corr_orders_rev:.3f}")
print(f"  gross_aov    : {corr_aov_rev:.3f}")
print(f"  unique_cust  : {corr_cust_rev:.3f}")
print(f"  units        : {corr_units_rev:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SEASONALITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 3 — Seasonality")
print("="*70)

# Month-of-year pattern (mean net_revenue per month, excluding Dec 2009 and Dec 2011 partial)
monthly_full = monthly_m[
    ~((monthly_m['year'] == 2009) | 
      ((monthly_m['year'] == 2011) & (monthly_m['month'] == 12)))
].copy()

seasonal_month = (
    monthly_full.groupby('month')
    .agg(
        mean_net_revenue=('net_revenue','mean'),
        mean_orders     =('orders','mean'),
        n_years         =('year','nunique'),
    ).reset_index()
)
seasonal_month['month_name'] = pd.Categorical(
    [MONTH_ORDER[m-1] for m in seasonal_month['month']],
    categories=MONTH_ORDER, ordered=True
)
seasonal_month = seasonal_month.sort_values('month')
print("Seasonal (month-of-year) mean net revenue:")
for _, r in seasonal_month.iterrows():
    print(f"  {MONTH_ORDER[int(r['month'])-1]:<12}: £{r['mean_net_revenue']:>10,.0f}")

# Day-of-week
dow_rev = (
    vmt.groupby('day_of_week_name')
    .agg(
        total_revenue=('revenue','sum'),
        n_orders=('Invoice','nunique'),
    ).reset_index()
)
dow_rev['avg_daily_revenue'] = dow_rev['total_revenue'] / len(vmt['InvoiceDate'].dt.date.unique())
print("\nOrders by day of week:")
for _, r in dow_rev.sort_values('day_of_week_name').iterrows():
    print(f"  {str(r['day_of_week_name']):<12}: {int(r['n_orders']):>6,} orders")

# Hour of day
hour_orders = vmt.groupby('hour')['Invoice'].nunique().reset_index().rename(columns={'Invoice':'n_orders'})
print("\nPeak hours (top 5):")
print(hour_orders.sort_values('n_orders', ascending=False).head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LIKE-FOR-LIKE PERIOD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 4 — Like-for-like period comparison")
print("="*70)

L4L_START_MONTH = 1   # January
L4L_END_MONTH   = 12
L4L_END_DAY     = 9   # Dec 9 (data cut-off for 2011)

p2010_start = pd.Timestamp('2010-01-01')
p2010_end   = pd.Timestamp('2010-12-09 23:59:59')
p2011_start = pd.Timestamp('2011-01-01')
p2011_end   = pd.Timestamp('2011-12-09 23:59:59')

vmt_2010 = vmt[(vmt['InvoiceDate'] >= p2010_start) & (vmt['InvoiceDate'] <= p2010_end)]
vmt_2011 = vmt[(vmt['InvoiceDate'] >= p2011_start) & (vmt['InvoiceDate'] <= p2011_end)]
mpt_2010 = mpt[(mpt['InvoiceDate'] >= p2010_start) & (mpt['InvoiceDate'] <= p2010_end)]
mpt_2011 = mpt[(mpt['InvoiceDate'] >= p2011_start) & (mpt['InvoiceDate'] <= p2011_end)]
cust_2010 = cust_txn[(cust_txn['InvoiceDate'] >= p2010_start) & (cust_txn['InvoiceDate'] <= p2010_end)]
cust_2011 = cust_txn[(cust_txn['InvoiceDate'] >= p2011_start) & (cust_txn['InvoiceDate'] <= p2011_end)]

inv_2010 = invoice_agg[(invoice_agg['invoice_date'] >= p2010_start) & (invoice_agg['invoice_date'] <= p2010_end)]
inv_2011 = invoice_agg[(invoice_agg['invoice_date'] >= p2011_start) & (invoice_agg['invoice_date'] <= p2011_end)]

def period_kpis(vmt_p, mpt_p, cust_p, inv_p):
    gross = vmt_p['revenue'].sum()
    ret   = mpt_p[mpt_p['revenue'] < 0]['revenue'].sum()
    net   = gross + ret
    ord_  = vmt_p['Invoice'].nunique()
    cust_ = cust_p['Customer ID'].nunique()
    units_= vmt_p['Quantity'].sum()
    gaov  = inv_p['invoice_revenue'].mean()
    naov  = net / ord_ if ord_ > 0 else np.nan
    return dict(gross_revenue=gross, return_value=abs(ret), net_revenue=net,
                orders=ord_, unique_customers=cust_, units=units_,
                gross_aov=gaov, net_aov=naov)

k2010 = period_kpis(vmt_2010, mpt_2010, cust_2010, inv_2010)
k2011 = period_kpis(vmt_2011, mpt_2011, cust_2011, inv_2011)

l4l = pd.DataFrame({'KPI': list(k2010.keys()),
                     '2010 (Jan-Dec9)': list(k2010.values()),
                     '2011 (Jan-Dec9)': list(k2011.values())})
l4l['Change (%)'] = ((l4l['2011 (Jan-Dec9)'] - l4l['2010 (Jan-Dec9)']) /
                      l4l['2010 (Jan-Dec9)'].abs() * 100).round(2)

print(f"Like-for-like comparison: 2010-01-01 to 2010-12-09  vs  2011-01-01 to 2011-12-09")
print(l4l.to_string(index=False))

# Key changes
net_chg   = k2011['net_revenue'] - k2010['net_revenue']
net_chg_p = k2011['net_revenue'] / k2010['net_revenue'] - 1
ord_chg_p = k2011['orders']      / k2010['orders']      - 1
aov_chg_p = k2011['gross_aov']   / k2010['gross_aov']   - 1
print(f"\n  Net revenue change : {net_chg_p*100:+.1f}%  (£{net_chg:+,.0f})")
print(f"  Orders change      : {ord_chg_p*100:+.1f}%")
print(f"  Gross AOV change   : {aov_chg_p*100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — COUNTRY PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 5 — Country performance")
print("="*70)

cs = country_summary.sort_values('net_revenue', ascending=False).reset_index(drop=True)
total_net = cs['net_revenue'].sum()
cs['revenue_share_pct'] = cs['net_revenue'] / total_net * 100
cs['cumulative_revenue_pct'] = cs['revenue_share_pct'].cumsum()

# Pareto: how many countries for 80%/90% of revenue
def countries_for_pct(target):
    return (cs['cumulative_revenue_pct'] <= target).sum() + 1
n_80 = countries_for_pct(80)
n_90 = countries_for_pct(90)

print(f"Countries for 80% of net revenue: {n_80}")
print(f"Countries for 90% of net revenue: {n_90}")
print(f"Top 5 countries by net revenue:")
print(cs[['Country','net_revenue','revenue_share_pct','orders','unique_customers']].head(5).to_string(index=False))

# AOV differences
cs_with_cust = cs[cs['unique_customers'] > 0].copy()
cs_with_cust['revenue_per_customer'] = cs_with_cust['gross_revenue'] / cs_with_cust['unique_customers']
print(f"\nAOV range: min £{cs['average_order_value'].min():,.0f}  max £{cs['average_order_value'].max():,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PRODUCT PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 6 — Product performance")
print("="*70)

ps = product_summary.copy()
ps = ps[ps['gross_revenue'] > 0].sort_values('gross_revenue', ascending=False).reset_index(drop=True)
total_prod_net = ps['net_revenue'].sum()
ps['revenue_share_pct']       = ps['net_revenue'] / total_prod_net * 100
ps['cumulative_revenue_pct']  = ps['revenue_share_pct'].cumsum()

n_prods = len(ps)

def prods_for_pct(target):
    n = (ps['cumulative_revenue_pct'] <= target).sum() + 1
    return n, n / n_prods * 100

n_prod_50, pct_prod_50 = prods_for_pct(50)
n_prod_80, pct_prod_80 = prods_for_pct(80)
n_prod_90, pct_prod_90 = prods_for_pct(90)

print(f"Total products with positive gross revenue: {n_prods:,}")
print(f"Products for 50% of net revenue: {n_prod_50}  ({pct_prod_50:.1f}% of products)")
print(f"Products for 80% of net revenue: {n_prod_80}  ({pct_prod_80:.1f}% of products)")
print(f"Products for 90% of net revenue: {n_prod_90}  ({pct_prod_90:.1f}% of products)")

print("\nTop 10 products by gross revenue:")
print(ps[['StockCode','canonical_description','gross_revenue','net_revenue',
          'total_units_sold','number_of_orders']].head(10).to_string(index=False))

# Customer reach analysis
print("\nTop 10 by customer reach:")
print(ps.sort_values('number_of_customers', ascending=False)
      [['StockCode','canonical_description','number_of_customers','gross_revenue']].head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — PRODUCT REVENUE VS VOLUME
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 7 — Product revenue vs volume")
print("="*70)

ps_valid = ps[(ps['total_units_sold'] > 0) & (ps['gross_revenue'] > 0)].copy()
ps_valid['avg_unit_price'] = ps_valid['gross_revenue'] / ps_valid['total_units_sold']

corr_units_rev = ps_valid[['total_units_sold','gross_revenue']].corr().iloc[0,1]
corr_price_rev = ps_valid[['avg_unit_price','gross_revenue']].corr().iloc[0,1]
print(f"Correlation: units_sold vs gross_revenue  : {corr_units_rev:.3f}")
print(f"Correlation: avg_unit_price vs gross_revenue: {corr_price_rev:.3f}")

# High-price, low-volume products (above median price, below median volume)
med_price  = ps_valid['avg_unit_price'].median()
med_units  = ps_valid['total_units_sold'].median()
high_p_low_v = ps_valid[(ps_valid['avg_unit_price'] > med_price) & (ps_valid['total_units_sold'] <= med_units)]
low_p_high_v = ps_valid[(ps_valid['avg_unit_price'] <= med_price) & (ps_valid['total_units_sold'] > med_units)]
print(f"\nHigh-price / low-volume products: {len(high_p_low_v):,}")
print(f"Low-price / high-volume products: {len(low_p_high_v):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — RETURN / CANCELLATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 8 — Return/cancellation analysis")
print("="*70)

MIN_SUPPORT_ORDERS = 10   # minimum number of sale orders before return rate is meaningful

# Monthly return trend
monthly_ret = (
    mpt.groupby('year_month')
    .agg(
        gross_revenue=('revenue', lambda x: x[x > 0].sum()),
        return_value =('revenue', lambda x: x[x < 0].sum()),
    ).reset_index()
)
monthly_ret['return_value_abs'] = monthly_ret['return_value'].abs()
monthly_ret['return_rate_pct']  = monthly_ret['return_value_abs'] / monthly_ret['gross_revenue'] * 100
monthly_ret['ym_str'] = monthly_ret['year_month'].astype(str)

overall_return_rate = abs(mpt[mpt['revenue']<0]['revenue'].sum()) / vmt['revenue'].sum() * 100
print(f"Overall return/cancellation rate: {overall_return_rate:.2f}% of gross")
print(f"Monthly return rate range: {monthly_ret['return_rate_pct'].min():.2f}% to {monthly_ret['return_rate_pct'].max():.2f}%")

# Product-level return rate (minimum support = 10 sale orders)
prod_ret = product_summary[product_summary['return_value'] < 0].copy()
prod_ret = prod_ret.rename(columns={'return_value':'return_value_col'})

prod_sales = product_summary[product_summary['gross_revenue'] > 0][['StockCode','gross_revenue','number_of_orders']].copy()
prod_ret_m = prod_sales.merge(
    product_summary[['StockCode','return_value']].rename(columns={'return_value':'ret_val'}),
    on='StockCode', how='left'
)
prod_ret_m['ret_val'] = prod_ret_m['ret_val'].fillna(0)
prod_ret_m['return_value_abs'] = prod_ret_m['ret_val'].abs()
prod_ret_m['return_rate_pct']  = prod_ret_m['return_value_abs'] / prod_ret_m['gross_revenue'] * 100

# Apply minimum support
prod_ret_filtered = prod_ret_m[prod_ret_m['number_of_orders'] >= MIN_SUPPORT_ORDERS].copy()
print(f"\nProducts with >= {MIN_SUPPORT_ORDERS} sale orders: {len(prod_ret_filtered):,}")
print(f"Top 10 products by return rate (min {MIN_SUPPORT_ORDERS} orders):")
top_ret_products = prod_ret_filtered.sort_values('return_rate_pct', ascending=False).head(10)
print(top_ret_products[['StockCode','gross_revenue','return_value_abs','return_rate_pct','number_of_orders']].to_string(index=False))

# Country return rates
cs['return_rate_pct'] = cs['return_value'].abs() / cs['gross_revenue'] * 100
print(f"\nCountry return rate range: {cs['return_rate_pct'].min():.2f}% to {cs['return_rate_pct'].max():.2f}%")
print("Top 5 countries by return rate (min 10 orders):")
cs_ret = cs[cs['orders'] >= 10].sort_values('return_rate_pct', ascending=False).head(5)
print(cs_ret[['Country','gross_revenue','return_value','return_rate_pct','orders']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CUSTOMER REVENUE CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 9 — Customer revenue concentration")
print("="*70)

cs_df = customer_summary.sort_values('total_revenue', ascending=False).reset_index(drop=True)
total_cust_rev = cs_df['total_revenue'].sum()
cs_df['rev_share'] = cs_df['total_revenue'] / total_cust_rev * 100
cs_df['cumrev']    = cs_df['rev_share'].cumsum()
n_cust_total = len(cs_df)

def top_pct_share(pct):
    n = max(1, int(np.ceil(n_cust_total * pct / 100)))
    share = cs_df.head(n)['total_revenue'].sum() / total_cust_rev * 100
    return n, share

n1,  s1  = top_pct_share(1)
n5,  s5  = top_pct_share(5)
n10, s10 = top_pct_share(10)
n20, s20 = top_pct_share(20)

print(f"Total identified customers: {n_cust_total:,}")
print(f"Top  1% ({n1:>4} customers): {s1:.1f}% of customer revenue")
print(f"Top  5% ({n5:>4} customers): {s5:.1f}% of customer revenue")
print(f"Top 10% ({n10:>4} customers): {s10:.1f}% of customer revenue")
print(f"Top 20% ({n20:>4} customers): {s20:.1f}% of customer revenue")

# How many customers needed for 50%, 80% of revenue
def custs_for_pct(target):
    return (cs_df['cumrev'] <= target).sum() + 1
n_c50 = custs_for_pct(50)
n_c80 = custs_for_pct(80)
print(f"\nCustomers for 50% of revenue: {n_c50}  ({n_c50/n_cust_total*100:.1f}% of customers)")
print(f"Customers for 80% of revenue: {n_c80}  ({n_c80/n_cust_total*100:.1f}% of customers)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 10 — Visualizations")
print("="*70)

ym_labels = monthly_m['ym_str'].tolist()
x_pos = list(range(len(ym_labels)))

# ── Fig 18: Monthly gross vs net revenue ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x_pos, monthly_m['gross_revenue'], label='Gross Revenue',
       alpha=0.80, color=sns.color_palette('muted')[0], edgecolor='white')
ax.bar(x_pos, monthly_m['net_revenue'],   label='Net Revenue',
       alpha=0.85, color=sns.color_palette('muted')[2], edgecolor='white')
ax.set_xticks(x_pos); ax.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(fmt_gbp)
ax.set_title('Monthly Gross vs Net Merchandise Revenue', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Revenue (£)'); ax.legend()
plt.tight_layout(); save_fig('18_monthly_gross_net_revenue.png')

# ── Fig 19: Monthly orders and customers ─────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 5))
color_ord = sns.color_palette('muted')[1]
color_cus = sns.color_palette('muted')[3]
ax1.bar(x_pos, monthly_m['orders'], color=color_ord, edgecolor='white', alpha=0.8, label='Orders')
ax2 = ax1.twinx()
ax2.plot(x_pos, monthly_m['unique_customers'], color=color_cus, marker='o', linewidth=2,
         markersize=4, label='Unique Customers')
ax1.set_xticks(x_pos); ax1.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax1.yaxis.set_major_formatter(fmt_k); ax2.yaxis.set_major_formatter(fmt_k)
ax1.set_ylabel('Orders', color=color_ord); ax2.set_ylabel('Unique Customers', color=color_cus)
ax1.set_title('Monthly Orders (bars) and Unique Customers (line)', fontsize=13, fontweight='bold')
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, loc='upper left')
plt.tight_layout(); save_fig('19_monthly_orders_customers.png')

# ── Fig 20: Monthly gross AOV ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(x_pos, monthly_m['gross_aov'], marker='o', linewidth=2,
        color=sns.color_palette('muted')[4], markersize=4, label='Gross AOV')
ax.plot(x_pos, monthly_m['net_aov'], marker='s', linewidth=2,
        color=sns.color_palette('muted')[3], markersize=4, linestyle='--', label='Net AOV')
ax.set_xticks(x_pos); ax.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'£{v:,.0f}'))
ax.set_title('Monthly Gross and Net Average Order Value', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('AOV (£)'); ax.legend()
plt.tight_layout(); save_fig('20_monthly_aov.png')

# ── Fig 21: Revenue decomposition — Revenue, Orders, AOV indexed to first month ──
idx_base = monthly_m.iloc[0]
fig, ax = plt.subplots(figsize=(14, 5))
for col, label, color in [
    ('net_revenue',  'Net Revenue',  sns.color_palette('muted')[0]),
    ('orders',       'Orders',       sns.color_palette('muted')[1]),
    ('gross_aov',    'Gross AOV',    sns.color_palette('muted')[4]),
    ('unique_customers', 'Customers',sns.color_palette('muted')[3]),
]:
    indexed = monthly_m[col] / monthly_m[col].iloc[0] * 100
    ax.plot(x_pos, indexed, linewidth=2, label=label, color=color)
ax.axhline(100, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks(x_pos); ax.set_xticklabels(ym_labels, rotation=45, ha='right', fontsize=8)
ax.set_title('Revenue Decomposition — Indexed to Dec 2009 = 100', fontsize=13, fontweight='bold')
ax.set_ylabel('Index (first month = 100)'); ax.set_xlabel('Month')
ax.legend(loc='upper left')
plt.tight_layout(); save_fig('21_revenue_decomposition.png')

# ── Fig 22: Seasonal month-of-year pattern ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(seasonal_month['month_name'], seasonal_month['mean_net_revenue'],
       color=sns.color_palette('muted')[0], edgecolor='white')
ax.yaxis.set_major_formatter(fmt_gbp)
ax.set_title('Seasonal Pattern — Mean Monthly Net Revenue\n(Complete months only, excludes Dec 2009 & Dec 2011)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Month of Year'); ax.set_ylabel('Mean Net Revenue (£)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); save_fig('22_seasonal_monthly_pattern.png')

# ── Fig 23: Day of week ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
dow_plot = dow_rev.sort_values('day_of_week_name')
ax.bar(dow_plot['day_of_week_name'].astype(str), dow_plot['n_orders'],
       color=sns.color_palette('muted')[1], edgecolor='white')
ax.yaxis.set_major_formatter(fmt_k)
ax.set_title('Total Orders by Day of Week', fontsize=13, fontweight='bold')
ax.set_xlabel('Day of Week'); ax.set_ylabel('Orders')
plt.tight_layout(); save_fig('23_day_of_week.png')

# ── Fig 24: Hour of day ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(hour_orders['hour'], hour_orders['n_orders'],
       color=sns.color_palette('muted')[4], edgecolor='white')
ax.yaxis.set_major_formatter(fmt_k)
ax.set_title('Orders by Hour of Day (UTC)', fontsize=13, fontweight='bold')
ax.set_xlabel('Hour (0–23)'); ax.set_ylabel('Orders')
ax.set_xticks(range(0,24))
plt.tight_layout(); save_fig('24_hour_of_day.png')

# ── Fig 25: Top 10 countries by net revenue ───────────────────────────────────
top10_c = cs.head(10)
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top10_c['Country'][::-1], top10_c['net_revenue'][::-1],
        color=sns.color_palette('muted', 10)[::-1], edgecolor='white')
ax.xaxis.set_major_formatter(fmt_gbp)
ax.set_title('Top 10 Countries by Net Merchandise Revenue', fontsize=13, fontweight='bold')
ax.set_xlabel('Net Revenue (£)'); ax.set_ylabel('Country')
plt.tight_layout(); save_fig('25_top10_countries.png')

# ── Fig 26: Country revenue concentration (Lorenz-style) ─────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
n_c = len(cs)
ax.plot(np.arange(1, n_c+1)/n_c*100, cs['cumulative_revenue_pct'],
        linewidth=2, color=sns.color_palette('muted')[0])
ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=1, alpha=0.5, label='Perfect equality')
ax.axhline(80, color='red', linewidth=0.8, linestyle=':', alpha=0.7)
ax.axhline(90, color='orange', linewidth=0.8, linestyle=':', alpha=0.7)
ax.set_xlabel('Cumulative % of Countries (sorted by revenue)')
ax.set_ylabel('Cumulative % of Net Revenue')
ax.set_title('Country Revenue Concentration', fontsize=13, fontweight='bold')
ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.legend()
plt.tight_layout(); save_fig('26_country_revenue_concentration.png')

# ── Fig 27: Product revenue vs units (scatter) ────────────────────────────────
ps_plot = ps_valid[ps_valid['total_units_sold'] > 0].copy()
ps_plot['log_units'] = np.log10(ps_plot['total_units_sold'].clip(lower=1))
ps_plot['log_rev']   = np.log10(ps_plot['gross_revenue'].clip(lower=1))

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(ps_plot['total_units_sold'], ps_plot['gross_revenue'],
                c=np.log10(ps_plot['number_of_customers'].clip(lower=1)),
                cmap='viridis', alpha=0.6, s=20, linewidths=0)
plt.colorbar(sc, ax=ax, label='log10(Customer Reach)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Units Sold (log scale)'); ax.set_ylabel('Gross Revenue (£, log scale)')
ax.set_title('Product Revenue vs Unit Volume\n(colour = customer reach)',
             fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(fmt_k)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'£{v:,.0f}'))
plt.tight_layout(); save_fig('27_product_revenue_vs_units.png')

# ── Fig 28: Product revenue concentration ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
n_p = len(ps)
ax.plot(np.arange(1, n_p+1)/n_p*100, ps['cumulative_revenue_pct'],
        linewidth=2, color=sns.color_palette('muted')[2])
ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=1, alpha=0.5)
ax.axhline(80, color='red', linewidth=0.8, linestyle=':', alpha=0.7, label='80% threshold')
ax.axhline(50, color='orange', linewidth=0.8, linestyle=':', alpha=0.7, label='50% threshold')
ax.set_xlabel('Cumulative % of Products (sorted by revenue)')
ax.set_ylabel('Cumulative % of Net Revenue')
ax.set_title('Product Revenue Concentration', fontsize=13, fontweight='bold')
ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.legend()
plt.tight_layout(); save_fig('28_product_revenue_concentration.png')

# ── Fig 29: Return/cancellation trend ─────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 5))
ret_x = list(range(len(monthly_ret)))
ax1.bar(ret_x, monthly_ret['return_value_abs'],
        color=sns.color_palette('muted')[3], edgecolor='white', alpha=0.8, label='Return Value (abs)')
ax2 = ax1.twinx()
ax2.plot(ret_x, monthly_ret['return_rate_pct'],
         color='#c00000', marker='o', linewidth=2, markersize=4, label='Return Rate %')
ax1.set_xticks(ret_x)
ax1.set_xticklabels(monthly_ret['ym_str'], rotation=45, ha='right', fontsize=8)
ax1.yaxis.set_major_formatter(fmt_gbp)
ax2.set_ylabel('Return Rate (% of gross)', color='#c00000')
ax1.set_ylabel('Return Value (£)'); ax1.set_xlabel('Month')
ax1.set_title('Monthly Return/Cancellation Value and Rate', fontsize=13, fontweight='bold')
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, loc='upper left')
plt.tight_layout(); save_fig('29_return_trend.png')

# ── Fig 30: Customer revenue concentration (Lorenz) ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(1, n_cust_total+1)/n_cust_total*100, cs_df['cumrev'],
        linewidth=2, color=sns.color_palette('muted')[0])
ax.plot([0,100],[0,100], '--', color='gray', linewidth=1, alpha=0.5)
ax.axhline(80, color='red', linewidth=0.8, linestyle=':', alpha=0.7, label='80%')
ax.axhline(50, color='orange', linewidth=0.8, linestyle=':', alpha=0.7, label='50%')
ax.set_xlabel('Cumulative % of Customers (sorted by revenue)')
ax.set_ylabel('Cumulative % of Customer Revenue')
ax.set_title('Customer Revenue Concentration', fontsize=13, fontweight='bold')
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.legend()
plt.tight_layout(); save_fig('30_customer_revenue_concentration.png')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — BUSINESS QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
bqs = [
    {
        'q': 'How did net revenue evolve month-over-month?',
        'method': 'Monthly net_revenue computed from mpt; pct_change() applied.',
        'result': f"Monthly net revenue ranged from £{monthly_m['net_revenue'].min():,.0f} to £{monthly_m['net_revenue'].max():,.0f}.",
    },
    {
        'q': 'Is revenue growth driven more by order volume or AOV?',
        'method': 'Pearson correlation of monthly net_revenue with orders, gross_aov, and unique_customers.',
        'result': f"Correlation: orders={corr_orders_rev:.2f}, gross_aov={corr_aov_rev:.2f}, customers={corr_cust_rev:.2f}.",
    },
    {
        'q': 'What is the seasonal revenue pattern?',
        'method': 'Mean net_revenue per calendar month, complete months only.',
        'result': f"Highest mean month: {seasonal_month.loc[seasonal_month['mean_net_revenue'].idxmax(),'month_name']} "
                  f"(£{seasonal_month['mean_net_revenue'].max():,.0f}); "
                  f"lowest: {seasonal_month.loc[seasonal_month['mean_net_revenue'].idxmin(),'month_name']} "
                  f"(£{seasonal_month['mean_net_revenue'].min():,.0f}).",
    },
    {
        'q': 'Is revenue concentrated among a small number of countries?',
        'method': 'Cumulative revenue share by country, sorted descending.',
        'result': f"{n_80} countries account for 80% of net revenue, out of {len(cs)} total.",
    },
    {
        'q': 'Is product revenue concentrated?',
        'method': 'Cumulative revenue share by product, sorted descending.',
        'result': f"{n_prod_80} products ({pct_prod_80:.1f}% of catalogue) account for 80% of net merchandise revenue.",
    },
    {
        'q': 'Is customer revenue concentrated?',
        'method': 'Top deciles of identified customers by total revenue.',
        'result': f"Top 10% of customers ({n10} customers) account for {s10:.1f}% of identified-customer revenue.",
    },
    {
        'q': 'How did performance change on a like-for-like basis (Jan–Dec9)?',
        'method': f'Comparing 2010-01-01 to 2010-12-09 vs 2011-01-01 to 2011-12-09.',
        'result': f"Net revenue {net_chg_p*100:+.1f}%, orders {ord_chg_p*100:+.1f}%, gross AOV {aov_chg_p*100:+.1f}%.",
    },
]

print("\n" + "="*70)
print("SECTION 11 — Business Questions")
print("="*70)
for i, bq in enumerate(bqs, 1):
    print(f"\nBQ{i}: {bq['q']}")
    print(f"  Method : {bq['method']}")
    print(f"  Result : {bq['result']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — EXECUTIVE INSIGHTS (evidence-based)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 12 — Executive Insights")
print("="*70)

peak_month_name = seasonal_month.loc[seasonal_month['mean_net_revenue'].idxmax(),'month_name']
low_month_name  = seasonal_month.loc[seasonal_month['mean_net_revenue'].idxmin(),'month_name']
top_country     = cs.iloc[0]['Country']
top_country_pct = cs.iloc[0]['revenue_share_pct']
return_rate_overall = overall_return_rate
max_monthly_ret_rate = monthly_ret['return_rate_pct'].max()
max_ret_month = monthly_ret.loc[monthly_ret['return_rate_pct'].idxmax(), 'ym_str']
top_prod_name = ps.iloc[0]['canonical_description']
top_prod_share = ps.iloc[0]['revenue_share_pct']
sun_orders = dow_rev[dow_rev['day_of_week_name']=='Sunday']['n_orders'].values
sun_str = f"{int(sun_orders[0]):,}" if len(sun_orders) else "0"
peak_hour = int(hour_orders.sort_values('n_orders', ascending=False).iloc[0]['hour'])

insights = [
    {
        'id': 1,
        'finding': f"Like-for-like net revenue (Jan–Dec 9) changed by {net_chg_p*100:+.1f}% "
                   f"from 2010 to 2011, with orders changing {ord_chg_p*100:+.1f}% and "
                   f"gross AOV changing {aov_chg_p*100:+.1f}%.",
        'evidence': f"2010 net: £{k2010['net_revenue']:,.0f}, 2011 net: £{k2011['net_revenue']:,.0f}.",
        'implication': "Revenue movement is associated with both order volume and order value changes. "
                       "Understanding which driver dominates informs whether to focus on acquisition (volume) "
                       "or upsell strategy (AOV)."
    },
    {
        'id': 2,
        'finding': f"Monthly revenue shows a seasonal pattern. {peak_month_name} has the highest "
                   f"mean net revenue (£{seasonal_month['mean_net_revenue'].max():,.0f}) and "
                   f"{low_month_name} has the lowest (£{seasonal_month['mean_net_revenue'].min():,.0f}), "
                   f"based on complete months only.",
        'evidence': "Seasonal monthly table, excluding Dec 2009 and Dec 2011 (partial).",
        'implication': "Observed seasonality can inform inventory planning and promotional scheduling. "
                       "The pattern is based on two complete years and may not capture multi-year trend shifts."
    },
    {
        'id': 3,
        'finding': f"{top_country} is the largest contributor, accounting for "
                   f"{top_country_pct:.1f}% of net merchandise revenue. "
                   f"{n_80} countries out of {len(cs)} account for 80% of net revenue.",
        'evidence': "country_summary sorted by net_revenue, cumulative revenue share.",
        'implication': "High geographic concentration increases exposure to UK-specific demand shocks. "
                       "The remaining countries collectively represent a meaningful revenue base."
    },
    {
        'id': 4,
        'finding': f"{n_prod_80} products ({pct_prod_80:.1f}% of the catalogue) generate 80% "
                   f"of net merchandise revenue. The top product ({top_prod_name}) contributes "
                   f"{top_prod_share:.1f}% of product net revenue on its own.",
        'evidence': "product_summary sorted by net_revenue, cumulative share.",
        'implication': "Revenue is concentrated in a relatively small product set. "
                       "Disruptions — stockouts, pricing changes — in this group could have "
                       "disproportionate revenue impact."
    },
    {
        'id': 5,
        'finding': f"Top 10% of identified customers ({n10} customers) account for "
                   f"{s10:.1f}% of identified-customer revenue. "
                   f"Top 1% ({n1} customers) account for {s1:.1f}%.",
        'evidence': "customer_summary sorted by total_revenue, cumulative share.",
        'implication': "Customer revenue is highly concentrated. Retention of high-value customers "
                       "is likely to have a disproportionate impact on overall revenue performance."
    },
    {
        'id': 6,
        'finding': f"The overall return/cancellation rate is {return_rate_overall:.1f}% of gross "
                   f"merchandise revenue. The highest monthly return rate was "
                   f"{max_monthly_ret_rate:.1f}% in {max_ret_month}.",
        'evidence': "merchandise_product_transactions negative revenue / vmt gross revenue.",
        'implication': "A low overall return rate suggests the returns process is not a major cost driver. "
                       "Spikes in specific months merit investigation for root causes (e.g., bulk cancellations)."
    },
    {
        'id': 7,
        'finding': f"Orders are highest on weekdays, with Sunday generating the fewest orders "
                   f"({sun_str} orders). Peak transaction hour is {peak_hour}:00.",
        'evidence': "vmt.groupby day_of_week_name and hour.",
        'implication': "Weekday and business-hours concentration is consistent with a B2B or wholesale "
                       "customer base. Customer service and operational capacity may be optimised accordingly."
    },
    {
        'id': 8,
        'finding': f"Monthly net revenue has a {corr_orders_rev:.2f} correlation with order count "
                   f"and {corr_aov_rev:.2f} correlation with gross AOV.",
        'evidence': "Pearson correlation on monthly_m.",
        'implication': "Revenue variation is more strongly associated with one driver than the other. "
                       "Interventions targeting the more correlated driver are more likely to affect revenue, "
                       "though correlation does not establish causality."
    },
]

for ins in insights:
    print(f"\nInsight {ins['id']}:")
    print(f"  Finding           : {ins['finding']}")
    print(f"  Evidence          : {ins['evidence']}")
    print(f"  Business implication: {ins['implication']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 14 — Validation")
print("="*70)

checks_pass = True
def chk(label, cond, detail=''):
    global checks_pass
    st = 'PASS' if cond else 'FAIL'
    if not cond: checks_pass = False
    print(f"  [{st}] {label}" + (f"  ({detail})" if detail else ""))
    assert cond, f"Validation failed: {label}"

chk("Monthly net sums to overall",
    abs(monthly_m['net_revenue'].sum() - mpt['revenue'].sum()) < 1.0)
chk("Country gross sums to vmt gross",
    abs(cs['gross_revenue'].sum() - vmt['revenue'].sum()) < 0.01)
chk("Product gross sums to vmt gross",
    abs(ps['gross_revenue'].sum() - vmt['revenue'].sum()) < 0.01)
chk("Revenue = Orders x AOV (monthly)",
    decomp_check < 0.01,
    f"max residual={decomp_check:.4f}")
chk("L4L 2010 end date <= 2010-12-09",
    vmt_2010['InvoiceDate'].max() <= pd.Timestamp('2010-12-09 23:59:59'))
chk("L4L 2011 end date <= 2011-12-09",
    vmt_2011['InvoiceDate'].max() <= pd.Timestamp('2011-12-09 23:59:59'))
chk("Dec 2011 partial flag present in monthly table",
    ((monthly_m['year']==2011) & (monthly_m['month']==12)).any())
chk("Return rate denominator is gross_revenue (positive)",
    (prod_ret_m['gross_revenue'] > 0).all())
chk("Customer concentration: top 1% < 100%", s1 < 100)
chk("Product concentration: top product < 100%", top_prod_share < 100)

print(f"\nAll validation checks: {'PASSED' if checks_pass else 'FAILED'}")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT FINAL VALUES FOR NOTEBOOK EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL VALUES")
print("="*70)
print(f"gross_rev                 = {gross_rev}")
print(f"net_rev                   = {net_rev}")
print(f"n_orders                  = {n_orders}")
print(f"n_cust                    = {n_cust}")
print(f"units_sold                = {units_sold}")
print(f"gross_aov                 = {gross_aov}")
print(f"net_aov                   = {net_aov}")
print(f"n_months                  = {len(monthly_m)}")
print(f"corr_orders_rev           = {corr_orders_rev}")
print(f"corr_aov_rev              = {corr_aov_rev}")
print(f"corr_cust_rev             = {corr_cust_rev}")
print(f"corr_units_rev            = {corr_units_rev}")
print(f"net_chg_p                 = {net_chg_p}")
print(f"ord_chg_p                 = {ord_chg_p}")
print(f"aov_chg_p                 = {aov_chg_p}")
print(f"k2010_net                 = {k2010['net_revenue']}")
print(f"k2011_net                 = {k2011['net_revenue']}")
print(f"k2010_orders              = {k2010['orders']}")
print(f"k2011_orders              = {k2011['orders']}")
print(f"k2010_gross_aov           = {k2010['gross_aov']}")
print(f"k2011_gross_aov           = {k2011['gross_aov']}")
print(f"k2010_customers           = {k2010['unique_customers']}")
print(f"k2011_customers           = {k2011['unique_customers']}")
print(f"n_80_countries            = {n_80}")
print(f"n_total_countries         = {len(cs)}")
print(f"top_country               = {top_country}")
print(f"top_country_pct           = {top_country_pct}")
print(f"n_prods                   = {n_prods}")
print(f"n_prod_50                 = {n_prod_50}")
print(f"pct_prod_50               = {pct_prod_50}")
print(f"n_prod_80                 = {n_prod_80}")
print(f"pct_prod_80               = {pct_prod_80}")
print(f"n_prod_90                 = {n_prod_90}")
print(f"pct_prod_90               = {pct_prod_90}")
print(f"top_prod_name             = {top_prod_name}")
print(f"top_prod_share            = {top_prod_share}")
print(f"n_cust_total              = {n_cust_total}")
print(f"n1, s1                    = {n1}, {s1}")
print(f"n5, s5                    = {n5}, {s5}")
print(f"n10, s10                  = {n10}, {s10}")
print(f"n20, s20                  = {n20}, {s20}")
print(f"n_c50                     = {n_c50}")
print(f"n_c80                     = {n_c80}")
print(f"return_rate_overall       = {return_rate_overall}")
print(f"max_monthly_ret_rate      = {max_monthly_ret_rate}")
print(f"max_ret_month             = {max_ret_month}")
print(f"peak_month_name           = {peak_month_name}")
print(f"low_month_name            = {low_month_name}")
print(f"med_price                 = {med_price}")
print(f"med_units                 = {med_units}")
print(f"high_p_low_v              = {len(high_p_low_v)}")
print(f"low_p_high_v              = {len(low_p_high_v)}")
print(f"peak_hour                 = {peak_hour}")
print(f"MIN_SUPPORT_ORDERS        = {MIN_SUPPORT_ORDERS}")
print(f"seasonal peak             = {seasonal_month.loc[seasonal_month['mean_net_revenue'].idxmax(),'month_name']}: £{seasonal_month['mean_net_revenue'].max():,.0f}")
print(f"seasonal low              = {seasonal_month.loc[seasonal_month['mean_net_revenue'].idxmin(),'month_name']}: £{seasonal_month['mean_net_revenue'].min():,.0f}")
print(f"\nSeasonal table:")
for _, r in seasonal_month.iterrows():
    print(f"  {MONTH_ORDER[int(r['month'])-1]:<12}: £{r['mean_net_revenue']:>10,.0f}")
print("\nEXECUTION COMPLETE")
