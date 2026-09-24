import pandas as pd
import numpy as np

cust = pd.read_csv('outputs/customer_summary.csv')
print('=== CUSTOMER INTELLIGENCE PREVIEW ===')
print()

# One-time vs repeat
one_time = (cust['total_orders'] == 1).sum()
repeat   = (cust['total_orders'] > 1).sum()
print(f'Total identified customers  : {len(cust):,}')
print(f'One-time customers          : {one_time:,} ({one_time/len(cust)*100:.1f}%)')
print(f'Repeat customers            : {repeat:,} ({repeat/len(cust)*100:.1f}%)')
print()

# Revenue distribution
total_rev = cust['total_revenue'].sum()
n1  = max(1, int(len(cust)*0.01))
n10 = max(1, int(len(cust)*0.10))
n25 = max(1, int(len(cust)*0.25))
top1  = cust.nlargest(n1,  'total_revenue')
top10 = cust.nlargest(n10, 'total_revenue')
top25 = cust.nlargest(n25, 'total_revenue')
print(f'Total customer revenue      : GBP {total_rev:,.2f}')
print(f'Top  1% ({n1:,} custs)  : GBP {top1["total_revenue"].sum():,.2f}  ({top1["total_revenue"].sum()/total_rev*100:.1f}%)')
print(f'Top 10% ({n10:,} custs)  : GBP {top10["total_revenue"].sum():,.2f}  ({top10["total_revenue"].sum()/total_rev*100:.1f}%)')
print(f'Top 25% ({n25:,} custs) : GBP {top25["total_revenue"].sum():,.2f}  ({top25["total_revenue"].sum()/total_rev*100:.1f}%)')
print()

# Order frequency distribution
print('Order frequency distribution:')
print(cust['total_orders'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
print()

# Revenue per customer distribution
print('Revenue per customer distribution:')
print(cust['total_revenue'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
print()

# AOV distribution
print('AOV distribution:')
print(cust['average_order_value'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
print()

# Customer lifetime
print('Customer lifetime (days):')
print(cust['customer_lifetime_days'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
print()

# Unique products per customer
print('Unique products per customer:')
print(cust['unique_products'].describe(percentiles=[.25,.5,.75,.90,.95,.99]))
print()

# n50 customers for 50% of revenue
cust_sorted = cust.sort_values('total_revenue', ascending=False).reset_index(drop=True)
cust_sorted['cum_pct'] = cust_sorted['total_revenue'].cumsum() / total_rev * 100
def n_for(t):
    return int((cust_sorted['cum_pct'] <= t).sum()) + 1

print('Customer revenue concentration:')
for t in [50, 75, 80, 90]:
    n = n_for(t)
    p = n / len(cust) * 100
    print(f'  {t}% of customer revenue -> {n} customers ({p:.1f}%)')
