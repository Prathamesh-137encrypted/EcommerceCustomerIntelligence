"""
Notebook 05 — Product & Geographic Intelligence
Execution script. Run from project root: python src/nb05_runner.py
"""
import os, re, sys, warnings, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore', message='.*data validation.*', category=UserWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.width', 160)

DATA_PATH   = 'data/online_retail_II.xlsx'
FIGURES_DIR = 'outputs/figures/'
OUTPUTS_DIR = 'outputs/'
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
DOW_ORDER   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

fmt_gbp = mticker.FuncFormatter(lambda v, _: f'£{v/1e6:.2f}M' if abs(v)>=1e6 else f'£{v:,.0f}')
fmt_k   = mticker.FuncFormatter(lambda v, _: f'{int(v):,}')

def save_fig(fname):
    p = os.path.join(FIGURES_DIR, fname)
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {p}')

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE (identical to NB02–04)
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

def assign_tt(row):
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

def modal_desc(s):
    nn = s.dropna(); return nn.mode().iloc[0] if len(nn) > 0 else None

print("Loading data...")
xl = pd.ExcelFile(DATA_PATH, engine='openpyxl')
tagged = []
for name in xl.sheet_names:
    df = pd.read_excel(DATA_PATH, sheet_name=name, engine='openpyxl',
                       dtype={'Invoice': str, 'StockCode': str})
    df['_sheet'] = name; tagged.append(df)
raw = pd.concat(tagged, ignore_index=True)
raw['InvoiceDate'] = pd.to_datetime(raw['InvoiceDate'], errors='coerce')
dup_mask = raw.duplicated(keep='first')
ct = raw[~dup_mask].copy().reset_index(drop=True)
ct['product_type']     = ct.apply(lambda r: classify_stockcode(r['StockCode'], r['Description']), axis=1)
ct['transaction_type'] = ct.apply(assign_tt, axis=1)
ct['revenue']          = ct['Quantity'] * ct['Price']
canon_map              = ct.groupby('StockCode')['Description'].agg(modal_desc)
ct['canonical_description'] = ct['StockCode'].map(canon_map)
vmt  = ct[(ct['transaction_type']=='SALE') & (ct['product_type']=='MERCHANDISE')].copy()
mpt  = ct[ct['product_type']=='MERCHANDISE'].copy()
ct_txn = vmt[vmt['Customer ID'].notna()].copy()
print(f"  vmt={len(vmt):,}  mpt={len(mpt):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PRODUCT ANALYTICAL TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 1 — Product analytical table\n"+"="*70)

prod_gross = (
    vmt.groupby('StockCode').agg(
        canonical_description=('canonical_description','first'),
        gross_revenue        =('revenue',     'sum'),
        positive_units       =('Quantity',    'sum'),
        orders               =('Invoice',     'nunique'),
        customers            =('Customer ID', 'nunique'),
    ).reset_index()
)

prod_neg = (
    mpt[mpt['revenue'] < 0].groupby('StockCode').agg(
        return_value  =('revenue',  'sum'),
        negative_units=('Quantity', 'sum'),
    ).reset_index()
)

prod_tbl = prod_gross.merge(prod_neg, on='StockCode', how='outer')
prod_tbl['canonical_description'] = prod_tbl['canonical_description'].fillna(prod_tbl['StockCode'].map(canon_map))
prod_tbl['return_value']   = prod_tbl['return_value'].fillna(0)
prod_tbl['negative_units'] = prod_tbl['negative_units'].fillna(0)
prod_tbl['gross_revenue']  = prod_tbl['gross_revenue'].fillna(0)
prod_tbl['positive_units'] = prod_tbl['positive_units'].fillna(0)
prod_tbl['orders']         = prod_tbl['orders'].fillna(0)
prod_tbl['customers']      = prod_tbl['customers'].fillna(0)
prod_tbl['net_revenue']    = prod_tbl['gross_revenue'] + prod_tbl['return_value']

# Derived metrics
prod_tbl['average_selling_price'] = np.where(
    prod_tbl['positive_units'] > 0,
    prod_tbl['gross_revenue'] / prod_tbl['positive_units'],
    np.nan
)
prod_tbl['revenue_per_order'] = np.where(
    prod_tbl['orders'] > 0,
    prod_tbl['gross_revenue'] / prod_tbl['orders'],
    np.nan
)
prod_tbl['revenue_per_customer'] = np.where(
    prod_tbl['customers'] > 0,
    prod_tbl['gross_revenue'] / prod_tbl['customers'],
    np.nan
)
prod_tbl['return_rate_pct'] = np.where(
    prod_tbl['gross_revenue'] > 0,
    prod_tbl['return_value'].abs() / prod_tbl['gross_revenue'] * 100,
    np.nan
)
prod_tbl['return_qty_ratio'] = np.where(
    prod_tbl['positive_units'] > 0,
    prod_tbl['negative_units'].abs() / prod_tbl['positive_units'] * 100,
    np.nan
)

# Active products (positive gross revenue)
prod_active = prod_tbl[prod_tbl['gross_revenue'] > 0].sort_values('gross_revenue', ascending=False).reset_index(drop=True)
n_active = len(prod_active)
print(f"Active products (gross > 0): {n_active:,}")

# Reconcile
gross_total = vmt['revenue'].sum()
prod_gross_sum = prod_active['gross_revenue'].sum()
assert abs(prod_gross_sum - gross_total) < 0.01, f"Product gross mismatch: {prod_gross_sum} vs {gross_total}"
print(f"Product gross reconciles: £{prod_gross_sum:,.2f} vs vmt £{gross_total:,.2f}")
mpt_net = mpt['revenue'].sum()
prod_net_sum = prod_tbl['net_revenue'].sum()
assert abs(prod_net_sum - mpt_net) < 0.01, f"Product net mismatch: {prod_net_sum} vs {mpt_net}"
print(f"Product net reconciles:  £{prod_net_sum:,.2f} vs mpt £{mpt_net:,.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PRODUCT PERFORMANCE RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 2 — Product performance rankings\n"+"="*70)

pa = prod_active.copy()
pa['net_rev_share']  = pa['net_revenue'] / pa['net_revenue'].sum() * 100
pa['cumul_net_rev']  = pa['net_rev_share'].cumsum()

# Top 15 by net revenue
print("\nTop 15 by NET revenue:")
print(pa.head(15)[['StockCode','canonical_description','gross_revenue','net_revenue','positive_units','orders','customers']].to_string(index=False))

# Top 15 by units
print("\nTop 15 by UNITS SOLD:")
print(pa.sort_values('positive_units', ascending=False).head(15)[
    ['StockCode','canonical_description','positive_units','gross_revenue','orders','customers']
].to_string(index=False))

# Top 15 by customer reach
print("\nTop 15 by CUSTOMER REACH:")
print(pa.sort_values('customers', ascending=False).head(15)[
    ['StockCode','canonical_description','customers','gross_revenue','positive_units']
].to_string(index=False))

# Top 15 by avg selling price (min 50 units to avoid noise)
pa_price = pa[pa['positive_units'] >= 50].sort_values('average_selling_price', ascending=False)
print("\nTop 15 by AVERAGE SELLING PRICE (min 50 units):")
print(pa_price.head(15)[['StockCode','canonical_description','average_selling_price','gross_revenue','positive_units']].to_string(index=False))

# Top 15 by revenue per customer
pa_rpc = pa[pa['customers'] >= 20].sort_values('revenue_per_customer', ascending=False)
print("\nTop 15 by REVENUE PER CUSTOMER (min 20 customers):")
print(pa_rpc.head(15)[['StockCode','canonical_description','revenue_per_customer','gross_revenue','customers']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PRODUCT REVENUE CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 3 — Product revenue concentration\n"+"="*70)

total_net = pa['net_revenue'].sum()
pa_sorted = pa.sort_values('net_revenue', ascending=False).reset_index(drop=True)
pa_sorted['cumul_net'] = pa_sorted['net_revenue'].cumsum()
pa_sorted['cumul_pct'] = pa_sorted['cumul_net'] / total_net * 100

def prods_for(target):
    n = int((pa_sorted['cumul_pct'] <= target).sum()) + 1
    return n, n / n_active * 100

n25, p25 = prods_for(25)
n50, p50 = prods_for(50)
n75, p75 = prods_for(75)
n80, p80 = prods_for(80)
n90, p90 = prods_for(90)

conc = {'25%': (n25, p25), '50%': (n50, p50), '75%': (n75, p75),
        '80%': (n80, p80), '90%': (n90, p90)}
print(f"Active products: {n_active:,}")
for tgt, (n, p) in conc.items():
    print(f"  Products for {tgt} of net revenue: {n:>5,}  ({p:.1f}% of catalogue)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PRODUCT VOLUME VS VALUE
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 4 — Product volume vs value\n"+"="*70)
pa_valid = pa[(pa['positive_units'] > 0) & (pa['net_revenue'] > 0)].copy()
corr_uv = pa_valid[['positive_units','net_revenue']].corr().iloc[0,1]
corr_pv = pa_valid[['average_selling_price','net_revenue']].dropna().corr().iloc[0,1]

med_units = pa_valid['positive_units'].median()
med_price = pa_valid['average_selling_price'].dropna().median()

print(f"Correlation units vs net_revenue      : {corr_uv:.3f}")
print(f"Correlation avg_price vs net_revenue  : {corr_pv:.3f}")
print(f"Median units per product              : {med_units:,.0f}")
print(f"Median avg selling price              : £{med_price:.2f}")

hv_hq = pa_valid[(pa_valid['positive_units'] > med_units) & (pa_valid['average_selling_price'].fillna(0) > med_price)]
hq_lv = pa_valid[(pa_valid['positive_units'] <= med_units) & (pa_valid['average_selling_price'].fillna(0) > med_price)]
lq_hv = pa_valid[(pa_valid['positive_units'] > med_units) & (pa_valid['average_selling_price'].fillna(0) <= med_price)]
print(f"\nHigh-volume, high-price products: {len(hv_hq)}")
print(f"High-price, low-volume products : {len(hq_lv)}")
print(f"High-volume, low-price products : {len(lq_hv)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PRODUCT RETURN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 5 — Product return analysis\n"+"="*70)

MIN_ORDERS = 10
MIN_UNITS  = 100

pa_ret = pa[pa['orders'] >= MIN_ORDERS].copy()
pa_ret_sorted = pa_ret.sort_values('return_rate_pct', ascending=False)
print(f"Products with >= {MIN_ORDERS} orders: {len(pa_ret):,}")

# Return rate distribution
print(f"\nReturn rate distribution (>= {MIN_ORDERS} orders):")
print(pa_ret['return_rate_pct'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))

print(f"\nTop 15 products by VALUE return rate (min {MIN_ORDERS} orders):")
print(pa_ret_sorted.head(15)[['StockCode','canonical_description',
    'gross_revenue','return_value','net_revenue','return_rate_pct','orders','positive_units']].to_string(index=False))

print(f"\nTop 15 products by QUANTITY return ratio (min {MIN_UNITS} positive units):")
pa_qty_ret = pa[pa['positive_units'] >= MIN_UNITS].sort_values('return_qty_ratio', ascending=False)
print(pa_qty_ret.head(15)[['StockCode','canonical_description',
    'positive_units','negative_units','return_qty_ratio','gross_revenue','orders']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — INVESTIGATE PRODUCT 23166
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 6 — Investigation: StockCode 23166\n"+"="*70)

rows_23166 = mpt[mpt['StockCode'] == '23166'].copy()
print(f"Total rows in mpt for 23166: {len(rows_23166)}")
print(f"Descriptions: {rows_23166['Description'].value_counts().to_dict()}")
print(f"\nTransaction type breakdown:")
print(rows_23166['transaction_type'].value_counts())

pos_23166 = rows_23166[rows_23166['revenue'] > 0]
neg_23166 = rows_23166[rows_23166['revenue'] < 0]
print(f"\nPositive rows : {len(pos_23166)} | gross revenue: £{pos_23166['revenue'].sum():,.2f}")
print(f"Negative rows : {len(neg_23166)} | return value : £{neg_23166['revenue'].sum():,.2f}")
print(f"Net revenue   : £{rows_23166['revenue'].sum():,.2f}")

print(f"\nPositive transaction details:")
print(pos_23166[['Invoice','InvoiceDate','Quantity','Price','revenue','Customer ID','Country']].to_string(index=False))

print(f"\nNegative transaction details:")
print(neg_23166[['Invoice','InvoiceDate','Quantity','Price','revenue','Customer ID','Country']].to_string(index=False))

# Check invoice prefix
print(f"\nInvoice prefixes for negative rows:")
print(neg_23166['Invoice'].apply(lambda x: str(x)[:1]).value_counts())
print(f"\nAll invoice numbers for 23166:")
print(rows_23166[['Invoice','InvoiceDate','Quantity','Price','revenue','transaction_type']].sort_values('InvoiceDate').to_string(index=False))

# Cross reference: do cancellation invoices match original invoices?
c_invoices_23166 = neg_23166['Invoice'].str.lstrip('C')
pos_invoice_list  = pos_23166['Invoice'].tolist()
matched = [ci for ci in c_invoices_23166 if ci in pos_invoice_list]
print(f"\nC-invoice numbers stripped: {list(c_invoices_23166)}")
print(f"Corresponding positive invoices found: {matched}")

# Investigate price of negative rows
print(f"\nPrice values in negative rows: {neg_23166['Price'].unique()}")
print(f"Quantity values in negative rows: {neg_23166['Quantity'].unique()}")
print(f"Price values in positive rows: {pos_23166['Price'].unique()}")
print(f"Quantity values in positive rows: {pos_23166['Quantity'].unique()}")

# Customer overlap
pos_custs_23166 = set(pos_23166['Customer ID'].dropna().unique())
neg_custs_23166 = set(neg_23166['Customer ID'].dropna().unique())
overlap = pos_custs_23166 & neg_custs_23166
print(f"\nDistinct customers on positive rows: {len(pos_custs_23166)}")
print(f"Distinct customers on negative rows: {len(neg_custs_23166)}")
print(f"Customer overlap (same cust on both): {len(overlap)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — INVESTIGATE PRODUCT 20879
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 7 — Investigation: StockCode 20879\n"+"="*70)

rows_20879 = mpt[mpt['StockCode'] == '20879'].copy()
print(f"Total rows in mpt for 20879: {len(rows_20879)}")
print(f"Descriptions: {rows_20879['Description'].value_counts().to_dict()}")
print(f"\nTransaction type breakdown:")
print(rows_20879['transaction_type'].value_counts())

pos_20879 = rows_20879[rows_20879['revenue'] > 0]
neg_20879 = rows_20879[rows_20879['revenue'] < 0]
print(f"\nPositive rows : {len(pos_20879)} | gross revenue: £{pos_20879['revenue'].sum():,.2f}")
print(f"Negative rows : {len(neg_20879)} | return value : £{neg_20879['revenue'].sum():,.2f}")
print(f"Net revenue   : £{rows_20879['revenue'].sum():,.2f}")

print(f"\nAll rows sorted by date:")
print(rows_20879[['Invoice','InvoiceDate','Quantity','Price','revenue',
                   'transaction_type','Customer ID','Country']].sort_values('InvoiceDate').to_string(index=False))

# Check for negative prices
neg_price_20879 = rows_20879[rows_20879['Price'] < 0]
print(f"\nRows with NEGATIVE PRICE: {len(neg_price_20879)}")
if len(neg_price_20879) > 0:
    print(neg_price_20879[['Invoice','Quantity','Price','revenue','transaction_type','Customer ID']].to_string(index=False))

# C-invoice matching
neg_rows_20879  = rows_20879[rows_20879['Quantity'] < 0]
c_inv_20879     = neg_rows_20879[neg_rows_20879['Invoice'].str.startswith('C')]
noc_inv_20879   = neg_rows_20879[~neg_rows_20879['Invoice'].str.startswith('C')]
print(f"\nNegative quantity rows: {len(neg_rows_20879)}")
print(f"  - with C prefix: {len(c_inv_20879)}")
print(f"  - without C prefix: {len(noc_inv_20879)}")
print(f"\nReturn rate calculation for 20879:")
g20879 = pos_20879['revenue'].sum()
r20879 = neg_20879['revenue'].sum()
print(f"  Gross: £{g20879:.2f}  Returns: £{r20879:.2f}  Return/Gross: {abs(r20879)/g20879*100:.1f}%")
print(f"  (Return > Gross because some negative rows have higher |value| than positive sales)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — COUNTRY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 8 — Country analysis\n"+"="*70)

country_gross = (
    vmt.groupby('Country').agg(
        gross_revenue=('revenue',   'sum'),
        orders       =('Invoice',   'nunique'),
        positive_units=('Quantity', 'sum'),
    ).reset_index()
)
country_cust = (
    ct_txn.groupby('Country')['Customer ID'].nunique()
    .reset_index().rename(columns={'Customer ID':'customers'})
)
country_ret = (
    mpt[mpt['revenue'] < 0].groupby('Country')['revenue'].sum()
    .reset_index().rename(columns={'revenue':'return_value'})
)

invoice_agg = (
    vmt.groupby('Invoice').agg(
        invoice_revenue=('revenue','sum'),
        invoice_units  =('Quantity','sum'),
        invoice_date   =('InvoiceDate','first'),
        country        =('Country','first'),
    ).reset_index()
)
country_aov = (
    invoice_agg.groupby('country').agg(
        gross_aov=('invoice_revenue','mean'),
        avg_units_per_order=('invoice_units','mean'),
    ).reset_index().rename(columns={'country':'Country'})
)

country_tbl = (
    country_gross
    .merge(country_cust, on='Country', how='left')
    .merge(country_ret,  on='Country', how='left')
    .merge(country_aov,  on='Country', how='left')
)
country_tbl['return_value']   = country_tbl['return_value'].fillna(0)
country_tbl['customers']      = country_tbl['customers'].fillna(0).astype(int)
country_tbl['net_revenue']    = country_tbl['gross_revenue'] + country_tbl['return_value']
country_tbl['net_aov']        = country_tbl['net_revenue'] / country_tbl['orders']
country_tbl['revenue_per_customer'] = np.where(
    country_tbl['customers'] > 0,
    country_tbl['gross_revenue'] / country_tbl['customers'], np.nan
)
country_tbl['return_rate_pct'] = (country_tbl['return_value'].abs() / country_tbl['gross_revenue'] * 100)
country_tbl = country_tbl.sort_values('net_revenue', ascending=False).reset_index(drop=True)
country_tbl['net_rev_share']  = country_tbl['net_revenue'] / country_tbl['net_revenue'].sum() * 100
country_tbl['cumul_net_share']= country_tbl['net_rev_share'].cumsum()

# Reconcile
assert abs(country_tbl['gross_revenue'].sum() - vmt['revenue'].sum()) < 0.01
print(f"Countries: {len(country_tbl)}")
print(country_tbl[['Country','gross_revenue','return_value','net_revenue',
                    'orders','customers','gross_aov','net_aov',
                    'revenue_per_customer','avg_units_per_order','return_rate_pct']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — UK VS INTERNATIONAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 9 — UK vs International\n"+"="*70)

def geo_group(c): return 'UK' if c == 'United Kingdom' else 'International'
country_tbl['geo_group'] = country_tbl['Country'].apply(geo_group)

geo = country_tbl.groupby('geo_group').agg(
    gross_revenue      =('gross_revenue','sum'),
    net_revenue        =('net_revenue',  'sum'),
    return_value       =('return_value', 'sum'),
    orders             =('orders',       'sum'),
    customers          =('customers',    'sum'),
    positive_units     =('positive_units','sum'),
    n_countries        =('Country',      'nunique'),
).reset_index()

# Invoice-level AOV per group
invoice_agg['geo_group'] = invoice_agg['country'].apply(geo_group)
geo_aov = invoice_agg.groupby('geo_group').agg(
    gross_aov=('invoice_revenue','mean'),
    net_aov  =('invoice_revenue', lambda x: x.sum()),
).reset_index()
# compute net_aov properly per group
for g in ['UK','International']:
    net_g  = country_tbl[country_tbl['geo_group']==g]['net_revenue'].sum()
    ord_g  = country_tbl[country_tbl['geo_group']==g]['orders'].sum()
    geo_aov.loc[geo_aov['geo_group']==g, 'net_aov'] = net_g / ord_g if ord_g > 0 else np.nan

geo = geo.merge(geo_aov[['geo_group','gross_aov','net_aov']], on='geo_group', how='left')
geo['revenue_per_customer'] = np.where(geo['customers']>0,
    geo['gross_revenue']/geo['customers'], np.nan)
geo['return_rate_pct'] = geo['return_value'].abs() / geo['gross_revenue'] * 100
geo['net_rev_share']   = geo['net_revenue'] / geo['net_revenue'].sum() * 100

print(geo[['geo_group','n_countries','gross_revenue','net_revenue','orders',
           'customers','gross_aov','net_aov','revenue_per_customer','return_rate_pct','net_rev_share']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — GEOGRAPHIC CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 10 — Geographic concentration\n"+"="*70)

total_cn = country_tbl['net_revenue'].sum()

def top_n_share(n):
    return country_tbl.head(n)['net_revenue'].sum() / total_cn * 100

def countries_for(target):
    return int((country_tbl['cumul_net_share'] <= target).sum()) + 1

s1c, s3c, s5c, s10c = top_n_share(1), top_n_share(3), top_n_share(5), top_n_share(10)
c50g, c75g, c80g, c90g = countries_for(50), countries_for(75), countries_for(80), countries_for(90)

print(f"Top  1 country  : {s1c:.1f}% of net revenue")
print(f"Top  3 countries: {s3c:.1f}%")
print(f"Top  5 countries: {s5c:.1f}%")
print(f"Top 10 countries: {s10c:.1f}%")
print(f"\nCountries needed for 50% net revenue: {c50g}")
print(f"Countries needed for 75% net revenue: {c75g}")
print(f"Countries needed for 80% net revenue: {c80g}")
print(f"Countries needed for 90% net revenue: {c90g}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — COUNTRY ECONOMICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 11 — Country economics\n"+"="*70)

MIN_CUST_C  = 10
MIN_ORDERS_C = 20

country_eco = country_tbl[
    (country_tbl['customers'] >= MIN_CUST_C) &
    (country_tbl['orders'] >= MIN_ORDERS_C)
].copy()
print(f"Countries with >= {MIN_CUST_C} customers and >= {MIN_ORDERS_C} orders: {len(country_eco)}")
print("\nBy gross AOV (desc):")
print(country_eco.sort_values('gross_aov', ascending=False)[
    ['Country','gross_aov','net_aov','revenue_per_customer','avg_units_per_order','orders','customers']
].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — PRODUCT × COUNTRY
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 12 — Product x Country analysis\n"+"="*70)

prod_country = (
    vmt.groupby(['StockCode','canonical_description','Country']).agg(
        net_revenue=('revenue',   'sum'),
        orders     =('Invoice',   'nunique'),
        customers  =('Customer ID','nunique'),
        units      =('Quantity',  'sum'),
    ).reset_index()
)

# Total product revenue (net) for share calculation
prod_net_total_map = prod_country.groupby('StockCode')['net_revenue'].sum()
prod_country['product_net_total'] = prod_country['StockCode'].map(prod_net_total_map)
prod_country['country_share_of_product'] = (
    prod_country['net_revenue'] / prod_country['product_net_total'] * 100
)

# Reconcile product-country to product level
pc_gross_sum = prod_country['net_revenue'].sum()
assert abs(pc_gross_sum - vmt['revenue'].sum()) < 0.01, f"PxC mismatch: {pc_gross_sum}"
print(f"Product-country rows: {len(prod_country):,}")

# Top 5 products per country (top 10 countries)
top10_countries = country_tbl.head(10)['Country'].tolist()
print("\nTop 5 products by net revenue, for each of the top 10 countries:")
for ctry in top10_countries:
    top5_ctry = (
        prod_country[prod_country['Country'] == ctry]
        .sort_values('net_revenue', ascending=False)
        .head(5)
    )
    print(f"\n  {ctry}:")
    print(top5_ctry[['StockCode','canonical_description','net_revenue','orders','customers']].to_string(index=False))

# Products with geographically concentrated revenue (>= 80% in one country)
prod_conc = (
    prod_country[prod_country['country_share_of_product'] >= 80]
    .merge(pa[['StockCode','gross_revenue']].rename(columns={'gross_revenue':'total_gross'}), on='StockCode', how='left')
)
prod_conc_filtered = prod_conc[prod_conc['total_gross'] >= 5000].sort_values('total_gross', ascending=False)
print(f"\nProducts with >= 80% of revenue in one country (min £5,000 total gross): {len(prod_conc_filtered)}")
print(prod_conc_filtered[['StockCode','canonical_description','Country',
    'net_revenue','country_share_of_product','total_gross']].head(20).to_string(index=False))

# Products with broad geographic reach (present in >= 15 countries)
prod_geo_reach = (
    prod_country.groupby('StockCode')['Country'].nunique()
    .reset_index().rename(columns={'Country':'n_countries'})
    .merge(pa[['StockCode','canonical_description','gross_revenue']], on='StockCode')
    .sort_values('n_countries', ascending=False)
)
print(f"\nProducts with broadest geographic reach:")
print(prod_geo_reach.head(15)[['StockCode','canonical_description','n_countries','gross_revenue']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — PRODUCT-MARKET OPPORTUNITY FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 13 — Product-market opportunity framework\n"+"="*70)

# Quadrant: High revenue + High customer reach
rev_med  = pa['gross_revenue'].median()
cust_med = pa['customers'].median()

pa['quadrant'] = np.select(
    [
        (pa['gross_revenue'] > rev_med)  & (pa['customers'] > cust_med),
        (pa['gross_revenue'] > rev_med)  & (pa['customers'] <= cust_med),
        (pa['gross_revenue'] <= rev_med) & (pa['customers'] > cust_med),
    ],
    ['HIGH_REV_HIGH_REACH', 'HIGH_REV_LOW_REACH', 'LOW_REV_HIGH_REACH'],
    default='LOW_REV_LOW_REACH'
)
print("Product quadrant distribution:")
print(pa['quadrant'].value_counts())
print("\nHigh revenue + High reach (top 15 by gross revenue):")
print(pa[pa['quadrant']=='HIGH_REV_HIGH_REACH'].head(15)[
    ['StockCode','canonical_description','gross_revenue','customers','average_selling_price','return_rate_pct']
].to_string(index=False))

# Internationalization candidates: high domestic (UK) share but decent absolute revenue
intl_cands = (
    prod_country[prod_country['Country'] != 'United Kingdom']
    .groupby(['StockCode','canonical_description'])[['net_revenue','orders','customers']].sum()
    .reset_index()
    .rename(columns={'net_revenue':'intl_revenue','orders':'intl_orders','customers':'intl_customers'})
    .merge(pa[['StockCode','gross_revenue']], on='StockCode')
)
intl_cands['intl_rev_share'] = intl_cands['intl_revenue'] / intl_cands['gross_revenue'] * 100
# High global revenue, low international share  = UK-concentrated opportunity
uk_conc_prods = intl_cands[
    (intl_cands['gross_revenue'] >= 10000) &
    (intl_cands['intl_rev_share'] <= 10)
].sort_values('gross_revenue', ascending=False).head(20)
print(f"\nHigh-revenue products with low international sales share (<= 10%, min £10k gross):")
print(uk_conc_prods[['StockCode','canonical_description','gross_revenue','intl_revenue','intl_rev_share']].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 14 — Visualizations\n"+"="*70)

# ── Fig 31: Top 15 products by net revenue ───────────────────────────────────
top15_r = pa.head(15)
fig, ax = plt.subplots(figsize=(12, 7))
labels = [f"{r['StockCode']}\n{str(r['canonical_description'])[:28]}" for _, r in top15_r.iterrows()]
ax.barh(labels[::-1], top15_r['net_revenue'][::-1],
        color=sns.color_palette('muted', 15)[::-1], edgecolor='white')
ax.xaxis.set_major_formatter(fmt_gbp)
ax.set_title('Top 15 Products by Net Merchandise Revenue', fontsize=12, fontweight='bold')
ax.set_xlabel('Net Revenue (£)'); ax.set_ylabel('Product')
plt.tight_layout(); save_fig('31_top15_products_net_revenue.png')

# ── Fig 32: Top 15 products by units ─────────────────────────────────────────
top15_u = pa.sort_values('positive_units', ascending=False).head(15)
fig, ax = plt.subplots(figsize=(12, 7))
labels_u = [f"{r['StockCode']}\n{str(r['canonical_description'])[:28]}" for _, r in top15_u.iterrows()]
ax.barh(labels_u[::-1], top15_u['positive_units'][::-1],
        color=sns.color_palette('muted', 15)[::-1], edgecolor='white')
ax.xaxis.set_major_formatter(fmt_k)
ax.set_title('Top 15 Products by Units Sold', fontsize=12, fontweight='bold')
ax.set_xlabel('Units Sold'); ax.set_ylabel('Product')
plt.tight_layout(); save_fig('32_top15_products_units.png')

# ── Fig 33: Product revenue concentration ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(1, n_active+1)/n_active*100, pa_sorted['cumul_pct'],
        linewidth=2, color=sns.color_palette('muted')[2])
ax.plot([0,100],[0,100], '--', color='gray', linewidth=1, alpha=0.5)
for tgt, col in [(50,'orange'),(80,'red'),(90,'darkred')]:
    ax.axhline(tgt, color=col, linewidth=0.8, linestyle=':', alpha=0.7, label=f'{tgt}%')
ax.set_xlabel('Cumulative % of Products (by revenue)'); ax.set_ylabel('Cumulative % of Net Revenue')
ax.set_title('Product Revenue Concentration (Pareto)', fontsize=12, fontweight='bold')
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.legend()
plt.tight_layout(); save_fig('33_product_revenue_concentration.png')

# ── Fig 34: Product revenue vs units scatter ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(pa_valid['positive_units'], pa_valid['net_revenue'],
                c=np.log10(pa_valid['customers'].clip(lower=1)),
                cmap='viridis', alpha=0.55, s=18, linewidths=0)
plt.colorbar(sc, ax=ax, label='log10(Customer Reach)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Units Sold (log)'); ax.set_ylabel('Net Revenue (£, log)')
ax.set_title('Product Revenue vs Unit Volume\n(colour = customer reach)',
             fontsize=12, fontweight='bold')
plt.tight_layout(); save_fig('34_product_revenue_vs_units.png')

# ── Fig 35: Product revenue vs customer reach ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
sc2 = ax.scatter(pa_valid['customers'], pa_valid['net_revenue'],
                 c=np.log10(pa_valid['positive_units'].clip(lower=1)),
                 cmap='plasma', alpha=0.55, s=18, linewidths=0)
plt.colorbar(sc2, ax=ax, label='log10(Units Sold)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of Customers (log)'); ax.set_ylabel('Net Revenue (£, log)')
ax.set_title('Product Revenue vs Customer Reach\n(colour = unit volume)',
             fontsize=12, fontweight='bold')
plt.tight_layout(); save_fig('35_product_revenue_vs_customers.png')

# ── Fig 36: Product return-rate distribution ─────────────────────────────────
pa_ret_hist = pa_ret[pa_ret['return_rate_pct'] > 0]['return_rate_pct'].clip(upper=100)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pa_ret_hist, bins=40, color=sns.color_palette('muted')[3], edgecolor='white', linewidth=0.4)
ax.axvline(pa_ret_hist.median(), color='red', linewidth=1.2, linestyle='--', label=f'Median {pa_ret_hist.median():.1f}%')
ax.yaxis.set_major_formatter(fmt_k)
ax.set_title(f'Product Return Rate Distribution (min {MIN_ORDERS} orders, capped at 100%)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Return Rate (% of gross, value-based)'); ax.set_ylabel('Products'); ax.legend()
plt.tight_layout(); save_fig('36_product_return_rate_distribution.png')

# ── Fig 37: Detailed return behavior — 23166 and 20879 ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, code, rows in [(axes[0], '23166', rows_23166), (axes[1], '20879', rows_20879)]:
    pos_r = rows[rows['revenue'] > 0]
    neg_r = rows[rows['revenue'] < 0]
    ax.bar(['Gross Sales\n(positive)','Returns\n(negative)'],
           [pos_r['revenue'].sum(), neg_r['revenue'].sum()],
           color=[sns.color_palette('muted')[2], sns.color_palette('muted')[3]],
           edgecolor='white')
    ax.yaxis.set_major_formatter(fmt_gbp)
    ax.set_title(f'StockCode {code}\n{str(rows["canonical_description"].iloc[0])[:35]}',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Revenue (£)')
plt.suptitle('Return Behavior Investigation: Products 23166 & 20879', fontsize=12, fontweight='bold')
plt.tight_layout(); save_fig('37_products_23166_20879_returns.png')

# ── Fig 38: Top 10 countries by net revenue ───────────────────────────────────
top10_c = country_tbl.head(10)
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top10_c['Country'][::-1], top10_c['net_revenue'][::-1],
        color=sns.color_palette('muted', 10)[::-1], edgecolor='white')
ax.xaxis.set_major_formatter(fmt_gbp)
ax.set_title('Top 10 Countries by Net Merchandise Revenue', fontsize=12, fontweight='bold')
ax.set_xlabel('Net Revenue (£)'); ax.set_ylabel('Country')
plt.tight_layout(); save_fig('38_top10_countries_net_revenue.png')

# ── Fig 39: Country revenue concentration ────────────────────────────────────
n_cntry = len(country_tbl)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(1,n_cntry+1)/n_cntry*100, country_tbl['cumul_net_share'],
        linewidth=2, color=sns.color_palette('muted')[0])
ax.plot([0,100],[0,100],'--',color='gray',linewidth=1,alpha=0.5)
ax.axhline(80, color='red', linewidth=0.8, linestyle=':',alpha=0.7, label='80%')
ax.axhline(90, color='orange', linewidth=0.8, linestyle=':',alpha=0.7, label='90%')
ax.set_xlabel('Cumulative % of Countries (by revenue)'); ax.set_ylabel('Cumulative % of Net Revenue')
ax.set_title('Country Revenue Concentration', fontsize=12, fontweight='bold')
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.legend()
plt.tight_layout(); save_fig('39_country_revenue_concentration.png')

# ── Fig 40: UK vs International ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col, title in [
    (axes[0], 'net_revenue',    'Net Revenue (£)'),
    (axes[1], 'orders',         'Orders'),
    (axes[2], 'revenue_per_customer', 'Revenue per Customer (£)'),
]:
    vals = [geo.loc[geo['geo_group']=='UK', col].values[0],
            geo.loc[geo['geo_group']=='International', col].values[0]]
    bars = ax.bar(['UK', 'International'], vals,
                   color=[sns.color_palette('muted')[0], sns.color_palette('muted')[1]], edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f'£{val:,.0f}' if 'Revenue' in title else f'{int(val):,}',
                ha='center', va='bottom', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
plt.suptitle('UK vs International Comparison', fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig('40_uk_vs_international.png')

# ── Fig 41: Country revenue per customer ─────────────────────────────────────
eco_plot = country_eco.sort_values('revenue_per_customer', ascending=False).head(15)
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(eco_plot['Country'][::-1], eco_plot['revenue_per_customer'][::-1],
        color=sns.color_palette('muted', len(eco_plot))[::-1], edgecolor='white')
ax.xaxis.set_major_formatter(fmt_gbp)
ax.set_title(f'Revenue per Customer by Country\n(min {MIN_CUST_C} customers, {MIN_ORDERS_C} orders)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Revenue per Customer (£)'); ax.set_ylabel('Country')
plt.tight_layout(); save_fig('41_country_revenue_per_customer.png')

# ── Fig 42: Product-country heatmap (top 15 products × top 10 countries) ────
top15_codes   = pa.head(15)['StockCode'].tolist()
top10_ctries  = country_tbl.head(10)['Country'].tolist()

heatmap_data = (
    prod_country[
        (prod_country['StockCode'].isin(top15_codes)) &
        (prod_country['Country'].isin(top10_ctries))
    ]
    .pivot_table(index='canonical_description', columns='Country', values='net_revenue', fill_value=0)
)
# Truncate description labels
heatmap_data.index = [str(i)[:35] for i in heatmap_data.index]
# Reorder columns to match country ranking
ordered_cols = [c for c in top10_ctries if c in heatmap_data.columns]
heatmap_data = heatmap_data[ordered_cols]

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    heatmap_data / 1000,
    ax=ax, cmap='YlOrRd', linewidths=0.3, linecolor='white',
    annot=True, fmt='.0f', annot_kws={'size': 7},
    cbar_kws={'label': 'Net Revenue (£000s)'}
)
ax.set_title('Product × Country Net Revenue Heatmap\n(Top 15 products × Top 10 countries, £000s)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Country'); ax.set_ylabel('Product')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.yticks(fontsize=7)
plt.tight_layout(); save_fig('42_product_country_heatmap.png')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — BUSINESS QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 15 — Business Questions\n"+"="*70)

bqs = [
    {
        'q': 'Which products contribute most to net revenue?',
        'method': 'prod_active sorted by net_revenue.',
        'result': f"Top product: {pa.iloc[0]['canonical_description']} (£{pa.iloc[0]['net_revenue']:,.0f} net). "
                  f"Top 3 together: £{pa.head(3)['net_revenue'].sum():,.0f}.",
    },
    {
        'q': 'How concentrated is product revenue?',
        'method': 'Cumulative net revenue share by product.',
        'result': f"{n50} products ({p50:.1f}%) -> 50% of net revenue. "
                  f"{n80} products ({p80:.1f}%) -> 80%.",
    },
    {
        'q': 'Are high-volume products also high-revenue products?',
        'method': 'Correlation between units and net revenue; quadrant analysis.',
        'result': f"Pearson r = {corr_uv:.3f}. {len(hv_hq)} products are simultaneously high-volume and high-price.",
    },
    {
        'q': 'Which countries contribute most revenue?',
        'method': 'country_tbl sorted by net_revenue.',
        'result': f"Top country: {country_tbl.iloc[0]['Country']} ({country_tbl.iloc[0]['net_rev_share']:.1f}%). "
                  f"Top 3: {s3c:.1f}%. Total countries: {len(country_tbl)}.",
    },
    {
        'q': 'Do international markets have different customer economics than the UK?',
        'method': 'UK vs International revenue_per_customer and gross_aov.',
        'result': f"UK revenue/customer: £{geo.loc[geo['geo_group']=='UK','revenue_per_customer'].values[0]:,.0f}. "
                  f"International: £{geo.loc[geo['geo_group']=='International','revenue_per_customer'].values[0]:,.0f}.",
    },
    {
        'q': 'Are there products concentrated in one geography?',
        'method': 'Products where one country accounts for >= 80% of revenue (min £5k gross).',
        'result': f"{len(prod_conc_filtered)} such products identified.",
    },
]

for i, bq in enumerate(bqs, 1):
    print(f"\nBQ{i}: {bq['q']}")
    print(f"  Method : {bq['method']}")
    print(f"  Result : {bq['result']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nSECTION 18 — Validation\n"+"="*70)

all_ok = True
def chk(label, cond, detail=''):
    global all_ok
    st = 'PASS' if cond else 'FAIL'
    if not cond: all_ok = False
    print(f"  [{st}] {label}" + (f"  ({detail})" if detail else ''))
    assert cond, f"Validation failed: {label}"

chk("Product gross sum == vmt gross",
    abs(prod_active['gross_revenue'].sum() - vmt['revenue'].sum()) < 0.01)
chk("Product net sum == mpt net",
    abs(prod_tbl['net_revenue'].sum() - mpt['revenue'].sum()) < 0.01)
chk("Country gross sum == vmt gross",
    abs(country_tbl['gross_revenue'].sum() - vmt['revenue'].sum()) < 0.01)
chk("UK + International net == total net",
    abs(geo['net_revenue'].sum() - country_tbl['net_revenue'].sum()) < 0.01)
chk("Product-country net sum == vmt net",
    abs(prod_country['net_revenue'].sum() - vmt['revenue'].sum()) < 0.01)
chk("Return rate denominator > 0 for filtered products",
    (pa_ret['gross_revenue'] > 0).all())
chk("23166 investigation: rows found",
    len(rows_23166) > 0)
chk("20879 investigation: rows found",
    len(rows_20879) > 0)
chk("Product concentration n80 < n_active",
    n80 < n_active)
chk("Geographic concentration top-1 < 100%",
    s1c < 100)
chk("UK + International orders == vmt orders",
    abs(geo['orders'].sum() - vmt['Invoice'].nunique()) < 1)

print(f"\nAll validation checks: {'PASSED' if all_ok else 'FAILED'}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"="*70+"\nFINAL VALUES\n"+"="*70)
print(f"n_active_products         = {n_active}")
print(f"n25, p25                  = {n25}, {p25:.2f}")
print(f"n50, p50                  = {n50}, {p50:.2f}")
print(f"n75, p75                  = {n75}, {p75:.2f}")
print(f"n80, p80                  = {n80}, {p80:.2f}")
print(f"n90, p90                  = {n90}, {p90:.2f}")
print(f"top_product_name          = {pa.iloc[0]['canonical_description']}")
print(f"top_product_gross         = {pa.iloc[0]['gross_revenue']:.2f}")
print(f"top_product_net           = {pa.iloc[0]['net_revenue']:.2f}")
print(f"top_product_sku           = {pa.iloc[0]['StockCode']}")
print(f"corr_uv                   = {corr_uv:.4f}")
print(f"corr_pv                   = {corr_pv:.4f}")
print(f"n_countries               = {len(country_tbl)}")
print(f"s1c, s3c, s5c, s10c       = {s1c:.2f}, {s3c:.2f}, {s5c:.2f}, {s10c:.2f}")
print(f"c50g, c75g, c80g, c90g    = {c50g}, {c75g}, {c80g}, {c90g}")
print(f"top_country               = {country_tbl.iloc[0]['Country']}")
print(f"top_country_share         = {country_tbl.iloc[0]['net_rev_share']:.2f}")
print(f"uk_net_rev                = {geo.loc[geo['geo_group']=='UK','net_revenue'].values[0]:.2f}")
print(f"intl_net_rev              = {geo.loc[geo['geo_group']=='International','net_revenue'].values[0]:.2f}")
print(f"uk_rev_per_cust           = {geo.loc[geo['geo_group']=='UK','revenue_per_customer'].values[0]:.2f}")
print(f"intl_rev_per_cust         = {geo.loc[geo['geo_group']=='International','revenue_per_customer'].values[0]:.2f}")
print(f"uk_gross_aov              = {geo.loc[geo['geo_group']=='UK','gross_aov'].values[0]:.2f}")
print(f"intl_gross_aov            = {geo.loc[geo['geo_group']=='International','gross_aov'].values[0]:.2f}")
print(f"prod_conc_filtered_count  = {len(prod_conc_filtered)}")
print(f"hv_hq_count               = {len(hv_hq)}")
print(f"uk_conc_prods_count       = {len(uk_conc_prods)}")
print(f"overall_return_rate       = {abs(mpt[mpt['revenue']<0]['revenue'].sum())/vmt['revenue'].sum()*100:.4f}")
# 23166 summary
print(f"\n23166 rows                = {len(rows_23166)}")
print(f"23166 pos rows            = {len(pos_23166)}, gross = £{pos_23166['revenue'].sum():.2f}")
print(f"23166 neg rows            = {len(neg_23166)}, return = £{neg_23166['revenue'].sum():.2f}")
print(f"23166 neg invoice prefixes= {neg_23166['Invoice'].apply(lambda x: str(x)[:1]).value_counts().to_dict()}")
print(f"23166 C-inv matched       = {len(matched)}")
# 20879 summary
print(f"\n20879 rows                = {len(rows_20879)}")
print(f"20879 pos rows            = {len(pos_20879)}, gross = £{pos_20879['revenue'].sum():.2f}")
print(f"20879 neg rows            = {len(neg_20879)}, return = £{neg_20879['revenue'].sum():.2f}")
print(f"20879 neg price rows      = {len(neg_price_20879)}")
print(f"20879 c_inv rows          = {len(c_inv_20879)}, noc_inv = {len(noc_inv_20879)}")
print("\nEXECUTION COMPLETE")
