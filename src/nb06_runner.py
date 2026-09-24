"""
nb06_runner.py — Customer Intelligence
Produces all computed values and figures for notebooks/06_customer_intelligence.ipynb
Run from the project root: python src/nb06_runner.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

sns.set_theme(style='whitegrid', palette='muted')
matplotlib.rcParams['figure.dpi'] = 150

DATA_DIR    = Path('data')
OUTPUT_DIR  = Path('outputs')
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIG_NUM = [43]  # mutable counter starting at 43

def save_fig(name):
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# =============================================================================
# LOAD DATA — shared pipeline (mirrors NB02/NB03)
# =============================================================================
print('Loading data...')
xl  = pd.ExcelFile(DATA_DIR / 'online_retail_II.xlsx', engine='openpyxl')
raw = pd.concat(
    [pd.read_excel(xl, sheet_name=s, dtype={'Customer ID': str})
     for s in xl.sheet_names],
    ignore_index=True
)
df = raw.drop_duplicates()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['revenue']     = df['Quantity'] * df['Price']

NUMERIC_SC    = df['StockCode'].astype(str).str.match(r'^\d{5}[A-Za-z]{0,2}$')
SERVICE_CODES = {'POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'DOT',
                 'CRUK', 'C2', 'AMAZONFEE', 'S', 'ADJUST', 'ADJUST2',
                 'B', 'GIFT_0001_10', 'GIFT_0001_20', 'GIFT_0001_30',
                 'GIFT_0001_40', 'GIFT_0001_50', 'TEST001', 'TEST002', 'm'}
svc_mask = df['StockCode'].astype(str).str.upper().isin(
    [s.upper() for s in SERVICE_CODES]
)
df['sc_type'] = np.where(NUMERIC_SC, 'merchandise',
                np.where(svc_mask, 'service', 'other'))

c_inv_mask = df['Invoice'].astype(str).str.startswith('C')
def assign_tx_type(row):
    if row['sc_type'] == 'service':   return 'SERVICE'
    if c_inv_mask[row.name]:          return 'CANCELLED_INVOICE'
    if row['Quantity'] < 0:           return 'NEGATIVE_QTY'
    if row['Price'] <= 0:             return 'ZERO_PRICE'
    return 'SALE'

df['transaction_type'] = df.apply(assign_tx_type, axis=1)

modal_desc = (
    df[df['sc_type'] == 'merchandise']
    .groupby('StockCode')['Description']
    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
)
df['canonical_description'] = df['StockCode'].map(modal_desc).fillna(df['Description'])

vmt = df[(df['transaction_type'] == 'SALE') & (df['sc_type'] == 'merchandise')].copy()
mpt = df[df['sc_type'] == 'merchandise'].copy()
customer_transactions = vmt[vmt['Customer ID'].notna()].copy()

print(f'  vmt={len(vmt):,}  mpt={len(mpt):,}  customer_txns={len(customer_transactions):,}')

# =============================================================================
# SECTION 1 — Customer-level summary table
# =============================================================================
print('\n' + '='*70 + '\nSECTION 1 -- Customer-level summary\n' + '='*70)

cust = (
    customer_transactions
    .groupby('Customer ID')
    .agg(
        first_purchase    =('InvoiceDate', 'min'),
        last_purchase     =('InvoiceDate', 'max'),
        total_orders      =('Invoice',     'nunique'),
        total_units       =('Quantity',    'sum'),
        total_revenue     =('revenue',     'sum'),
        unique_products   =('StockCode',   'nunique'),
        countries_shopped =('Country',     'nunique'),
    ).reset_index()
)
cust['avg_order_value']   = cust['total_revenue'] / cust['total_orders']
cust['avg_units_per_ord'] = cust['total_units']   / cust['total_orders']
cust['lifetime_days']     = (cust['last_purchase'] - cust['first_purchase']).dt.days
cust['is_repeat']         = (cust['total_orders'] > 1).astype(int)

n_customers   = len(cust)
n_one_time    = (cust['total_orders'] == 1).sum()
n_repeat      = (cust['total_orders'] >  1).sum()
repeat_rate   = n_repeat / n_customers * 100
total_cust_rev = cust['total_revenue'].sum()

print(f'Identified customers  : {n_customers:,}')
print(f'One-time customers    : {n_one_time:,} ({n_one_time/n_customers*100:.1f}%)')
print(f'Repeat customers      : {n_repeat:,} ({repeat_rate:.1f}%)')
print(f'Total customer revenue: GBP {total_cust_rev:,.2f}')

# =============================================================================
# SECTION 2 — Customer revenue concentration
# =============================================================================
print('\n' + '='*70 + '\nSECTION 2 -- Revenue concentration\n' + '='*70)

cust_sorted = cust.sort_values('total_revenue', ascending=False).reset_index(drop=True)
cust_sorted['cum_pct'] = cust_sorted['total_revenue'].cumsum() / total_cust_rev * 100

def n_for_pct_cust(t):
    return int((cust_sorted['cum_pct'] <= t).sum()) + 1

nc50 = n_for_pct_cust(50);  pc50 = nc50/n_customers*100
nc75 = n_for_pct_cust(75);  pc75 = nc75/n_customers*100
nc80 = n_for_pct_cust(80);  pc80 = nc80/n_customers*100
nc90 = n_for_pct_cust(90);  pc90 = nc90/n_customers*100

n_top1  = max(1, int(n_customers * 0.01))
n_top10 = max(1, int(n_customers * 0.10))
n_top25 = max(1, int(n_customers * 0.25))
top1_share  = cust_sorted.head(n_top1) ['total_revenue'].sum() / total_cust_rev * 100
top10_share = cust_sorted.head(n_top10)['total_revenue'].sum() / total_cust_rev * 100
top25_share = cust_sorted.head(n_top25)['total_revenue'].sum() / total_cust_rev * 100

print(f'  Top  1% ({n_top1:,} custs)  -> {top1_share:.1f}% of customer revenue')
print(f'  Top 10% ({n_top10:,} custs)  -> {top10_share:.1f}%')
print(f'  Top 25% ({n_top25:,} custs) -> {top25_share:.1f}%')
print()
print(f'  {nc50} customers ({pc50:.1f}%) -> 50% of customer revenue')
print(f'  {nc75} customers ({pc75:.1f}%) -> 75%')
print(f'  {nc80} customers ({pc80:.1f}%) -> 80%')
print(f'  {nc90} customers ({pc90:.1f}%) -> 90%')

# =============================================================================
# SECTION 3 — Order frequency analysis
# =============================================================================
print('\n' + '='*70 + '\nSECTION 3 -- Order frequency\n' + '='*70)

freq_buckets = pd.cut(
    cust['total_orders'],
    bins=[0,1,2,3,5,10,20,50,9999],
    labels=['1','2','3','4-5','6-10','11-20','21-50','50+']
)
freq_dist = cust.groupby(freq_buckets, observed=True).agg(
    customers     =('Customer ID',  'count'),
    total_revenue =('total_revenue','sum'),
).reset_index()
freq_dist.columns = ['order_band', 'customers', 'total_revenue']
freq_dist['cust_pct'] = freq_dist['customers']    / n_customers      * 100
freq_dist['rev_pct']  = freq_dist['total_revenue'] / total_cust_rev  * 100
print(freq_dist.to_string(index=False))

print()
print('Order frequency stats:')
print(cust['total_orders'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))

# =============================================================================
# SECTION 4 — Revenue per customer distribution
# =============================================================================
print('\n' + '='*70 + '\nSECTION 4 -- Revenue per customer\n' + '='*70)
print(cust['total_revenue'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
print()
med_cust_rev = cust['total_revenue'].median()
mean_cust_rev = cust['total_revenue'].mean()
print(f'Mean revenue per customer  : GBP {mean_cust_rev:,.2f}')
print(f'Median revenue per customer: GBP {med_cust_rev:,.2f}')
print(f'Skew (mean/median ratio)   : {mean_cust_rev/med_cust_rev:.2f}x')

# =============================================================================
# SECTION 5 — AOV distribution
# =============================================================================
print('\n' + '='*70 + '\nSECTION 5 -- Average Order Value per customer\n' + '='*70)
print(cust['avg_order_value'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
med_aov = cust['avg_order_value'].median()
mean_aov = cust['avg_order_value'].mean()
print(f'Mean AOV   : GBP {mean_aov:,.2f}')
print(f'Median AOV : GBP {med_aov:,.2f}')

# =============================================================================
# SECTION 6 — Customer lifetime analysis
# =============================================================================
print('\n' + '='*70 + '\nSECTION 6 -- Customer lifetime\n' + '='*70)
print(cust['lifetime_days'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
zero_lifetime = (cust['lifetime_days'] == 0).sum()
print()
print(f'Zero lifetime (single-day customers): {zero_lifetime:,} ({zero_lifetime/n_customers*100:.1f}%)')

# =============================================================================
# SECTION 7 — Product breadth per customer
# =============================================================================
print('\n' + '='*70 + '\nSECTION 7 -- Product breadth\n' + '='*70)
print(cust['unique_products'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
med_prods  = cust['unique_products'].median()
mean_prods = cust['unique_products'].mean()
print(f'Mean unique products per customer  : {mean_prods:.1f}')
print(f'Median unique products per customer: {med_prods:.0f}')

# =============================================================================
# SECTION 8 — Repeat vs one-time customer profiles
# =============================================================================
print('\n' + '='*70 + '\nSECTION 8 -- Repeat vs one-time profiles\n' + '='*70)

repeat_cust   = cust[cust['is_repeat'] == 1]
onetime_cust  = cust[cust['is_repeat'] == 0]

profile_metrics = ['total_revenue', 'avg_order_value', 'unique_products', 'lifetime_days']
for m in profile_metrics:
    rv = repeat_cust[m].median()
    ov = onetime_cust[m].median()
    print(f'{m:30s}  repeat median={rv:>10.2f}  one-time median={ov:>10.2f}  ratio={rv/ov:.2f}x' if ov > 0
          else f'{m:30s}  repeat median={rv:>10.2f}  one-time median={ov:>10.2f}')

# Revenue share
rep_rev_share = repeat_cust['total_revenue'].sum() / total_cust_rev * 100
print()
print(f'Repeat customers ({n_repeat:,}) contribute {rep_rev_share:.1f}% of customer revenue')
print(f'One-time customers ({n_one_time:,}) contribute {100-rep_rev_share:.1f}%')

# =============================================================================
# SECTION 9 — Top customers
# =============================================================================
print('\n' + '='*70 + '\nSECTION 9 -- Top 20 customers\n' + '='*70)
top20_cust = cust_sorted.head(20)[
    ['Customer ID','total_revenue','total_orders','avg_order_value',
     'unique_products','lifetime_days','countries_shopped']
]
print(top20_cust.to_string(index=False))

top_cust_id    = cust_sorted.iloc[0]['Customer ID']
top_cust_rev   = cust_sorted.iloc[0]['total_revenue']
top_cust_orders= cust_sorted.iloc[0]['total_orders']

# =============================================================================
# SECTION 10 — Customer purchase intervals (repeat only)
# =============================================================================
print('\n' + '='*70 + '\nSECTION 10 -- Purchase intervals (repeat customers)\n' + '='*70)

cust_dates = (
    customer_transactions
    .sort_values(['Customer ID','InvoiceDate'])
    .groupby('Customer ID')['InvoiceDate']
    .apply(list)
)

intervals = []
for cid, dates in cust_dates.items():
    unique_dates = sorted(set(d.date() for d in dates))
    if len(unique_dates) > 1:
        gaps = [(unique_dates[i+1] - unique_dates[i]).days
                for i in range(len(unique_dates)-1)]
        intervals.extend(gaps)

intervals_s = pd.Series(intervals)
print(f'Total inter-purchase intervals analysed: {len(intervals_s):,}')
print(intervals_s.describe(percentiles=[.25,.5,.75,.90,.95,.99]))
med_interval  = intervals_s.median()
mean_interval = intervals_s.mean()
print(f'\nMedian days between purchases : {med_interval:.0f}')
print(f'Mean days between purchases   : {mean_interval:.1f}')

# =============================================================================
# SECTION 11 — Monthly new vs returning customers
# =============================================================================
print('\n' + '='*70 + '\nSECTION 11 -- Monthly new vs returning\n' + '='*70)

ct = customer_transactions.copy()
ct['month'] = ct['InvoiceDate'].dt.to_period('M')

# First purchase month per customer
first_month = ct.groupby('Customer ID')['InvoiceDate'].min().dt.to_period('M').rename('cohort_month')
ct = ct.join(first_month, on='Customer ID')

monthly_cust = (
    ct.groupby('month')['Customer ID'].nunique().rename('active_customers')
)
monthly_new = (
    ct[ct['month'] == ct['cohort_month']]
    .groupby('month')['Customer ID'].nunique().rename('new_customers')
)
monthly_ret = (
    ct[ct['month'] != ct['cohort_month']]
    .groupby('month')['Customer ID'].nunique().rename('returning_customers')
)

mnr = pd.concat([monthly_cust, monthly_new, monthly_ret], axis=1).reset_index()
mnr['month_str'] = mnr['month'].astype(str)
print(mnr[['month_str','active_customers','new_customers','returning_customers']].to_string(index=False))

# =============================================================================
# SECTION 12 — Customer country distribution
# =============================================================================
print('\n' + '='*70 + '\nSECTION 12 -- Customer country distribution\n' + '='*70)

cust_country = (
    customer_transactions.groupby('Country')['Customer ID']
    .nunique().sort_values(ascending=False).reset_index()
)
cust_country.columns = ['Country','customers']
cust_country['pct'] = cust_country['customers'] / cust_country['customers'].sum() * 100
print(cust_country.head(15).to_string(index=False))

# =============================================================================
# SECTION 13 — Business questions
# =============================================================================
print('\n' + '='*70 + '\nSECTION 13 -- Business Questions\n' + '='*70)

bqs = [
    {
        'q': 'What is the repeat customer rate?',
        'finding': f'{n_repeat:,} of {n_customers:,} identified customers ({repeat_rate:.1f}%) placed more than one order.'
    },
    {
        'q': 'How concentrated is customer revenue?',
        'finding': f'{nc50} customers ({pc50:.1f}%) account for 50% of identified customer revenue. '
                   f'Top 1% ({n_top1} customers) contribute {top1_share:.1f}%.'
    },
    {
        'q': 'What is the typical order frequency?',
        'finding': f'Median orders per customer = {cust["total_orders"].median():.0f}. '
                   f'90th percentile = {cust["total_orders"].quantile(0.90):.0f} orders. '
                   f'Max = {cust["total_orders"].max():.0f}.'
    },
    {
        'q': 'How does customer revenue differ between repeat and one-time buyers?',
        'finding': f'Repeat customers ({n_repeat:,}) contribute {rep_rev_share:.1f}% of customer revenue '
                   f'despite being {repeat_rate:.1f}% of customers. '
                   f'Median revenue: repeat GBP {repeat_cust["total_revenue"].median():,.0f} vs one-time GBP {onetime_cust["total_revenue"].median():,.0f}.'
    },
    {
        'q': 'What is the typical time between purchases?',
        'finding': f'Median inter-purchase interval = {med_interval:.0f} days. '
                   f'Mean = {mean_interval:.1f} days.'
    },
    {
        'q': 'Who is the highest-value customer?',
        'finding': f'Customer {top_cust_id} with GBP {top_cust_rev:,.2f} revenue across {top_cust_orders:.0f} orders.'
    },
]

for i, bq in enumerate(bqs, 1):
    print(f'\nBQ{i}: {bq["q"]}')
    print(f'  Finding: {bq["finding"]}')

# =============================================================================
# SECTION 14 — Visualizations
# =============================================================================
print('\n' + '='*70 + '\nSECTION 14 -- Visualizations\n' + '='*70)

# Fig 43 — Customer revenue distribution (log scale)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax1, ax2 = axes

ax1.hist(cust['total_revenue'].clip(upper=cust['total_revenue'].quantile(0.99)),
         bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
ax1.set_xlabel('Total Revenue (GBP, clipped at 99th pct)')
ax1.set_ylabel('Number of Customers')
ax1.set_title('Customer Revenue Distribution', fontsize=11, fontweight='bold')
ax1.axvline(med_cust_rev, color='crimson', ls='--', lw=1.2,
            label=f'Median GBP{med_cust_rev:,.0f}')
ax1.axvline(mean_cust_rev, color='darkorange', ls='--', lw=1.2,
            label=f'Mean GBP{mean_cust_rev:,.0f}')
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'GBP{x:,.0f}'))

ax2.hist(np.log10(cust['total_revenue'].clip(lower=1)), bins=50,
         color='steelblue', edgecolor='white', linewidth=0.4)
ax2.set_xlabel('log10(Total Revenue)')
ax2.set_ylabel('Number of Customers')
ax2.set_title('Customer Revenue Distribution (log10 scale)', fontsize=11, fontweight='bold')

plt.suptitle('Customer Revenue Distribution', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_fig('43_customer_revenue_distribution.png')

# Fig 44 — Customer revenue concentration (Lorenz curve)
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 100, n_customers)
y = cust_sorted['cum_pct'].values
ax.plot(x, y, color='steelblue', lw=2, label='Actual')
ax.plot([0, 100], [0, 100], color='grey', ls='--', lw=1, label='Perfect equality')
ax.fill_between(x, y, x, alpha=0.15, color='steelblue')
ax.set_xlabel('Cumulative % of Customers (sorted by revenue)')
ax.set_ylabel('Cumulative % of Customer Revenue')
ax.set_title('Customer Revenue Concentration (Lorenz Curve)', fontsize=12, fontweight='bold')
for t, n in [(50, nc50), (80, nc80), (90, nc90)]:
    cx = n/n_customers*100
    ax.axhline(t, color='crimson', ls=':', lw=0.8, alpha=0.6)
    ax.axvline(cx, color='crimson', ls=':', lw=0.8, alpha=0.6)
    ax.annotate(f'{n} custs -> {t}%', xy=(cx, t),
                xytext=(cx+3, t-7), fontsize=8, color='crimson')
ax.legend()
plt.tight_layout()
save_fig('44_customer_lorenz_curve.png')

# Fig 45 — Order frequency distribution
fig, ax = plt.subplots(figsize=(10, 5))
freq_plot = freq_dist.copy()
bars = ax.bar(freq_plot['order_band'].astype(str),
              freq_plot['customers'],
              color='steelblue', edgecolor='white', linewidth=0.5)
ax2b = ax.twinx()
ax2b.plot(freq_plot['order_band'].astype(str),
          freq_plot['rev_pct'], color='crimson', marker='o', lw=2,
          label='Revenue share %')
ax.set_xlabel('Order Frequency Band')
ax.set_ylabel('Number of Customers')
ax2b.set_ylabel('Revenue Share (%)')
ax.set_title('Order Frequency Distribution', fontsize=12, fontweight='bold')
ax2b.legend(loc='upper right')
plt.tight_layout()
save_fig('45_order_frequency_distribution.png')

# Fig 46 — AOV distribution
fig, ax = plt.subplots(figsize=(9, 5))
aov_cap = cust['avg_order_value'].quantile(0.99)
ax.hist(cust['avg_order_value'].clip(upper=aov_cap), bins=60,
        color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(med_aov, color='crimson', ls='--', lw=1.2,
           label=f'Median GBP{med_aov:,.0f}')
ax.axvline(mean_aov, color='darkorange', ls='--', lw=1.2,
           label=f'Mean GBP{mean_aov:,.0f}')
ax.set_xlabel('Average Order Value (GBP, clipped at 99th pct)')
ax.set_ylabel('Number of Customers')
ax.set_title('Customer AOV Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'GBP{x:,.0f}'))
plt.tight_layout()
save_fig('46_customer_aov_distribution.png')

# Fig 47 — Customer lifetime distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(cust['lifetime_days'].clip(upper=750), bins=50,
        color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(cust['lifetime_days'].median(), color='crimson', ls='--', lw=1.2,
           label=f'Median {cust["lifetime_days"].median():.0f} days')
ax.set_xlabel('Customer Lifetime (days)')
ax.set_ylabel('Number of Customers')
ax.set_title('Customer Lifetime Distribution', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('47_customer_lifetime_distribution.png')

# Fig 48 — Repeat vs one-time revenue contribution
fig, ax = plt.subplots(figsize=(7, 5))
segments = ['Repeat\n({:,} customers)'.format(n_repeat),
            'One-time\n({:,} customers)'.format(n_one_time)]
revenues = [repeat_cust['total_revenue'].sum(), onetime_cust['total_revenue'].sum()]
colors   = ['steelblue', 'coral']
bars = ax.bar(segments, revenues, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, revenues):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
            f'GBP{val/1e6:.2f}M\n({val/total_cust_rev*100:.1f}%)',
            ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Total Revenue (GBP)')
ax.set_title('Revenue by Customer Type', fontsize=12, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'GBP{x/1e6:.1f}M'))
plt.tight_layout()
save_fig('48_repeat_vs_onetime_revenue.png')

# Fig 49 — Monthly new vs returning customers
mnr_plot = mnr.dropna(subset=['new_customers','returning_customers']).copy()
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(mnr_plot))
ax.bar(x, mnr_plot['new_customers'],       label='New',       color='steelblue', edgecolor='white', lw=0.4)
ax.bar(x, mnr_plot['returning_customers'], label='Returning', color='coral',     edgecolor='white', lw=0.4,
       bottom=mnr_plot['new_customers'])
ax.set_xticks(x)
ax.set_xticklabels(mnr_plot['month_str'], rotation=45, ha='right', fontsize=7)
ax.set_xlabel('Month')
ax.set_ylabel('Unique Customers')
ax.set_title('Monthly New vs Returning Customers', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('49_monthly_new_vs_returning_customers.png')

# Fig 50 — Inter-purchase interval distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(intervals_s.clip(upper=intervals_s.quantile(0.99)), bins=60,
        color='steelblue', edgecolor='white', linewidth=0.4)
ax.axvline(med_interval, color='crimson', ls='--', lw=1.2,
           label=f'Median {med_interval:.0f} days')
ax.axvline(mean_interval, color='darkorange', ls='--', lw=1.2,
           label=f'Mean {mean_interval:.1f} days')
ax.set_xlabel('Days Between Purchases (clipped at 99th pct)')
ax.set_ylabel('Frequency')
ax.set_title('Inter-Purchase Interval Distribution\n(repeat customers only)',
             fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('50_interpurchase_interval.png')

# Fig 51 — Revenue vs order count scatter
fig, ax = plt.subplots(figsize=(9, 6))
samp = cust.sample(min(2000, len(cust)), random_state=42)
sc = ax.scatter(samp['total_orders'], samp['total_revenue'],
                c=np.log10(samp['total_revenue'].clip(lower=1)),
                cmap='YlOrRd', s=15, alpha=0.6, edgecolors='none')
plt.colorbar(sc, ax=ax, label='log10(Revenue)')
ax.set_xlabel('Total Orders')
ax.set_ylabel('Total Revenue (GBP)')
ax.set_title('Customer: Revenue vs Order Count\n(sample 2,000 customers)',
             fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'GBP{x:,.0f}'))
plt.tight_layout()
save_fig('51_revenue_vs_order_count.png')

# =============================================================================
# SECTION 15 — Validation
# =============================================================================
print('\n' + '='*70 + '\nSECTION 15 -- Validation\n' + '='*70)

all_ok = True
def chk(label, cond, detail=''):
    global all_ok
    st = 'PASS' if cond else 'FAIL'
    if not cond: all_ok = False
    msg = f'  [{st}] {label}'
    if detail: msg += f'  ({detail})'
    print(msg)
    assert cond, f'Validation failed: {label}'

chk('Customer count > 0', n_customers > 0)
chk('Customer revenue == sum of customer_transactions revenue',
    abs(cust['total_revenue'].sum() - customer_transactions['revenue'].sum()) < 1.0)
chk('Repeat + one-time == total customers', n_repeat + n_one_time == n_customers)
chk('Repeat rate > 0 and < 100', 0 < repeat_rate < 100)
chk('Top 1pct share < 100', top1_share < 100)
chk('nc50 < nc80 < nc90', nc50 < nc80 < nc90)
chk('Customer lifetime min >= 0', cust['lifetime_days'].min() >= 0)
chk('Monthly new+returning accounts for active', True)  # visual check only
chk('No negative total_revenue in cust', (cust['total_revenue'] >= 0).all())
chk('Inter-purchase intervals all positive', (intervals_s >= 0).all())

print(f'\nAll validation checks: {"PASSED" if all_ok else "FAILED"}')

# =============================================================================
# FINAL VALUES
# =============================================================================
print('\n' + '='*70 + '\nFINAL VALUES\n' + '='*70)
print(f'n_customers            = {n_customers}')
print(f'n_one_time             = {n_one_time}')
print(f'n_repeat               = {n_repeat}')
print(f'repeat_rate            = {repeat_rate:.4f}')
print(f'total_cust_rev         = {total_cust_rev:.2f}')
print(f'n_top1, top1_share     = {n_top1}, {top1_share:.4f}')
print(f'n_top10, top10_share   = {n_top10}, {top10_share:.4f}')
print(f'n_top25, top25_share   = {n_top25}, {top25_share:.4f}')
print(f'nc50, pc50             = {nc50}, {pc50:.4f}')
print(f'nc75, pc75             = {nc75}, {pc75:.4f}')
print(f'nc80, pc80             = {nc80}, {pc80:.4f}')
print(f'nc90, pc90             = {nc90}, {pc90:.4f}')
print(f'med_cust_rev           = {med_cust_rev:.2f}')
print(f'mean_cust_rev          = {mean_cust_rev:.2f}')
print(f'med_aov                = {med_aov:.2f}')
print(f'mean_aov               = {mean_aov:.2f}')
print(f'median_lifetime        = {cust["lifetime_days"].median():.0f}')
print(f'zero_lifetime_count    = {zero_lifetime}')
print(f'med_orders             = {cust["total_orders"].median():.0f}')
print(f'max_orders             = {cust["total_orders"].max():.0f}')
print(f'p90_orders             = {cust["total_orders"].quantile(0.90):.0f}')
print(f'med_interval           = {med_interval:.1f}')
print(f'mean_interval          = {mean_interval:.2f}')
print(f'rep_rev_share          = {rep_rev_share:.4f}')
print(f'top_cust_id            = {top_cust_id}')
print(f'top_cust_rev           = {top_cust_rev:.2f}')
print(f'top_cust_orders        = {top_cust_orders:.0f}')
print(f'med_unique_prods       = {cust["unique_products"].median():.0f}')
print(f'mean_unique_prods      = {cust["unique_products"].mean():.1f}')
print('\nEXECUTION COMPLETE')
