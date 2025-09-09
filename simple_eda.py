import pandas as pd
import numpy as np

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - SME LOAN DATASET")
print("=" * 60)

# Load the data
df = pd.read_csv('improved_sme_loan_data2004.csv')
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Basic info
print(f"\nData Types:")
print(df.dtypes)

# Missing values
print(f"\nMissing Values:")
missing_data = df.isnull().sum()
for col, missing in missing_data.items():
    if missing > 0:
        print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")

# Numerical features summary
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumerical Features Summary:")
print(df[numerical_cols].describe())

# Categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nCategorical Features Summary:")

for col in categorical_cols:
    print(f"\n{col.upper()}:")
    value_counts = df[col].value_counts()
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Top 5 values:")
    for i, (value, count) in enumerate(value_counts.head().items(), 1):
        print(f"    {i}. {value}: {count} ({count/len(df)*100:.1f}%)")

# Target variable analysis
print(f"\nTARGET VARIABLE ANALYSIS:")
if 'loan_approved' in df.columns:
    print(f"Loan Approved Distribution:")
    approved_counts = df['loan_approved'].value_counts()
    for value, count in approved_counts.items():
        print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

if 'risk_level' in df.columns:
    print(f"\nRisk Level Distribution:")
    risk_counts = df['risk_level'].value_counts()
    for value, count in risk_counts.items():
        print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

# Business insights
print(f"\nBUSINESS INSIGHTS:")

if 'monthly_revenue' in df.columns:
    print(f"Revenue Analysis:")
    print(f"  Average: ₹{df['monthly_revenue'].mean():,.2f}")
    print(f"  Median: ₹{df['monthly_revenue'].median():,.2f}")
    print(f"  Min: ₹{df['monthly_revenue'].min():,.2f}")
    print(f"  Max: ₹{df['monthly_revenue'].max():,.2f}")

if 'credit_score' in df.columns:
    print(f"\nCredit Score Analysis:")
    print(f"  Average: {df['credit_score'].mean():.2f}")
    print(f"  Median: {df['credit_score'].median():.2f}")
    print(f"  Min: {df['credit_score'].min()}")
    print(f"  Max: {df['credit_score'].max()}")

if 'business_age' in df.columns:
    print(f"\nBusiness Age Analysis:")
    print(f"  Average: {df['business_age'].mean():.2f} years")
    print(f"  Median: {df['business_age'].median():.2f} years")
    print(f"  Min: {df['business_age'].min()} years")
    print(f"  Max: {df['business_age'].max()} years")

if 'gst_compliance' in df.columns:
    print(f"\nGST Compliance Analysis:")
    gst_counts = df['gst_compliance'].value_counts()
    for value, count in gst_counts.items():
        print(f"  {value}: {count} ({count/len(df)*100:.1f}%)")

if 'business_type' in df.columns:
    print(f"\nBusiness Sector Analysis:")
    sector_counts = df['business_type'].value_counts()
    print("  Top 5 sectors:")
    for i, (sector, count) in enumerate(sector_counts.head().items(), 1):
        print(f"    {i}. {sector}: {count} ({count/len(df)*100:.1f}%)")

# Correlation analysis
numerical_df = df.select_dtypes(include=[np.number])
if len(numerical_df.columns) > 1:
    print(f"\nCORRELATION ANALYSIS:")
    correlation_matrix = numerical_df.corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))

# Data quality
print(f"\nDATA QUALITY ASSESSMENT:")
duplicates = df.duplicated().sum()
print(f"  Duplicate rows: {duplicates}")
print(f"  Data quality: {'Good' if duplicates == 0 and missing_data.sum() == 0 else 'Needs attention'}")

print("\n" + "=" * 60)
print("EDA COMPLETED!")
print("=" * 60) 