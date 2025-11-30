import pandas as pd

# 1) Load dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

print("Dataset loaded successfully!")

# 2) Basic info
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 3) Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4) Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"\nRemoved {before - after} duplicate rows")

# 5) Convert Order Date to datetime safely
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month

# --- FEATURE ENGINEERING (copy this block after cleaning & date conversion) ---
import numpy as np

# 1) Profit margin (safe: avoid divide-by-zero)
if 'Sales' in df.columns and 'Profit' in df.columns:
    df['ProfitMargin'] = df['Profit'] / df['Sales']
    df['ProfitMargin'] = df['ProfitMargin'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 2) Order Year / Month name / Day-of-week
if 'Order Date' in df.columns:
    df['OrderYear'] = df['Order Date'].dt.year
    df['OrderMonth'] = df['Order Date'].dt.month
    df['OrderMonthName'] = df['Order Date'].dt.strftime('%b')
    df['OrderDay'] = df['Order Date'].dt.day
    df['OrderWeekday'] = df['Order Date'].dt.day_name()

# 3) Delivery time in days (Ship Date - Order Date)
if 'Ship Date' in df.columns and 'Order Date' in df.columns:
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
    df['DeliveryDays'] = (df['Ship Date'] - df['Order Date']).dt.days
    # negative or NaN delivery days make sense to review
    df['DeliveryDays'] = df['DeliveryDays'].fillna(-1)  # -1 means missing

# 4) Revenue per unit (Sales / Quantity)
if 'Sales' in df.columns and 'Quantity' in df.columns:
    df['RevPerUnit'] = df['Sales'] / df['Quantity']
    df['RevPerUnit'] = df['RevPerUnit'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 5) High discount flag
if 'Discount' in df.columns:
    df['HighDiscountFlag'] = (df['Discount'] > 0.3).astype(int)

# 6) Customer-level aggregates (total sales, orders, avg order value)
if 'Customer ID' in df.columns:
    cust_sales = df.groupby('Customer ID')['Sales'].sum().rename('cust_total_sales')
    cust_orders = df.groupby('Customer ID')['Order ID'].nunique().rename('cust_total_orders')
    # merge back
    df = df.merge(cust_sales, on='Customer ID', how='left')
    df = df.merge(cust_orders, on='Customer ID', how='left')
    df['cust_avg_order_value'] = df['cust_total_sales'] / df['cust_total_orders']
    df['RepeatCustomerFlag'] = (df['cust_total_orders'] > 1).astype(int)

# 7) Simple label/one-hot for Category (small number of unique values)
if 'Category' in df.columns:
    # If you want dummies (one-hot) - uncomment below
    # dummies = pd.get_dummies(df['Category'], prefix='cat')
    # df = pd.concat([df, dummies], axis=1)

    # Or just label encode (for quick ML)
    df['Category_Label'] = df['Category'].astype('category').cat.codes

# 8) Recency (days since last order) - if you want a global 'today'
# set a reference date (e.g., last order in dataset or today)
ref_date = df['Order Date'].max()
df['DaysSinceOrder'] = (ref_date - df['Order Date']).dt.days

# 9) Save engineered dataset for later use
df.to_csv("retail_with_features.csv", index=False)
print("Feature engineering done. Saved file: retail_with_features.csv")
# --- end feature block ---

# 6) Total Sales & Profit
if 'Sales' in df.columns:
    print("\nTotal Sales:", df['Sales'].sum())
if 'Profit' in df.columns:
    print("Total Profit:", df['Profit'].sum())

# 7) Top Categories
if 'Sub-Category' in df.columns:
    print("\nTop Sub-Categories by Sales:")
    print(df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10))

# 8) Region-wise Sales
if 'Region' in df.columns:
    print("\nSales by Region:")
    print(df.groupby('Region')['Sales'].sum())

# ====== VISUALIZATIONS (save plots) ======
import matplotlib.pyplot as plt
import seaborn as sns
import os

# make folder for plots
os.makedirs("plots", exist_ok=True)

# 1) Top 10 sub-categories (bar)
if 'Sub-Category' in df.columns:
    top10 = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top10.values, y=top10.index)
    plt.title("Top 10 Sub-Categories by Sales")
    plt.xlabel("Sales")
    plt.tight_layout()
    plt.savefig("plots/top10_subcategories.png")
    plt.close()

# 2) Monthly sales trend (Year-Month)
if 'Order Date' in df.columns:
    df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('YearMonth')['Sales'].sum().reset_index()
    plt.figure(figsize=(12,5))
    sns.lineplot(data=monthly, x='YearMonth', y='Sales', marker='o')
    plt.xticks(rotation=45)
    plt.title("Monthly Sales Trend")
    plt.tight_layout()
    plt.savefig("plots/monthly_sales_trend.png")
    plt.close()

# 3) Sales by Region (pie)
if 'Region' in df.columns:
    region_sales = df.groupby('Region')['Sales'].sum()
    plt.figure(figsize=(6,6))
    plt.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', startangle=140)
    plt.title("Sales by Region")
    plt.tight_layout()
    plt.savefig("plots/sales_by_region.png")
    plt.close()

# 4) Sales vs Profit scatter (to find low-profit high-sales)
if 'Sales' in df.columns and 'Profit' in df.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df.sample(1000, random_state=1), x='Sales', y='Profit', alpha=0.6)  # sample for speed
    plt.title("Sales vs Profit (sample)")
    plt.tight_layout()
    plt.savefig("plots/sales_vs_profit.png")
    plt.close()

print("\nPlots saved in folder: plots/ (top10_subcategories.png, monthly_sales_trend.png, sales_by_region.png, sales_vs_profit.png)")


#Terminal me full dataset print karna
#print(df)
#Fir run karo:
#python analysis.py

#Sirf top 10 rows dekhna (best practice)
#print(df.head(10))

#Sirf last 10 rows dekhna
#print(df.tail(10))

#Sirf columns dekhna
#print(df.columns)

#Sirf numbers ka summary (mean, min, max)
#print(df.describe())
