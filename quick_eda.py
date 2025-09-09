import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('improved_sme_loan_data2004.csv')

print("=" * 60)
print("SME LOAN DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\nColumns: {list(df.columns)}")

print(f"\nData Types:")
print(df.dtypes)

print(f"\nMissing Values:")
missing = df.isnull().sum()
for col, count in missing.items():
    if count > 0:
        print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")

print(f"\nNumerical Features Summary:")
numerical_cols = ['monthly_revenue', 'credit_score', 'bank_balance', 'business_age', 'risk_score', 'max_loan_amount', 'interest_rate']
print(df[numerical_cols].describe())

print(f"\nTarget Variable - Loan Approved:")
approved_counts = df['loan_approved'].value_counts()
for value, count in approved_counts.items():
    print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

print(f"\nRisk Level Distribution:")
risk_counts = df['risk_level'].value_counts()
for value, count in risk_counts.items():
    print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

print(f"\nBusiness Type Distribution:")
business_counts = df['business_type'].value_counts()
for i, (business, count) in enumerate(business_counts.head(10).items(), 1):
    print(f"  {i}. {business}: {count} ({count/len(df)*100:.1f}%)")

print(f"\nGST Compliance:")
gst_counts = df['gst_compliance'].value_counts()
for value, count in gst_counts.items():
    print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

print(f"\nBusiness Insights:")
print(f"  Average Monthly Revenue: ₹{df['monthly_revenue'].mean():,.2f}")
print(f"  Average Credit Score: {df['credit_score'].mean():.2f}")
print(f"  Average Business Age: {df['business_age'].mean():.2f} years")
print(f"  Average Bank Balance: ₹{df['bank_balance'].mean():,.2f}")

print(f"\nCorrelation with Loan Approval:")
# Convert boolean to int for correlation
df_corr = df.copy()
df_corr['loan_approved'] = df_corr['loan_approved'].astype(int)
df_corr['gst_compliance'] = df_corr['gst_compliance'].astype(int)

# Select only numerical columns for correlation
numerical_df = df_corr[['monthly_revenue', 'credit_score', 'bank_balance', 'business_age', 'risk_score', 'max_loan_amount', 'interest_rate', 'loan_approved', 'gst_compliance']]
correlation = numerical_df.corr()['loan_approved'].sort_values(ascending=False)
for feature, corr in correlation.items():
    if feature != 'loan_approved':
        print(f"  {feature}: {corr:.3f}")

print("\n" + "=" * 60) 