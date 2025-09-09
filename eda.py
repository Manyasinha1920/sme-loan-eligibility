import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - SME LOAN DATASET")
print("=" * 60)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# 1. Load the data
print("\n1. LOADING DATA")
print("-" * 30)
df = pd.read_csv('improved_sme_loan_data2004.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# 2. Basic Information
print("\n2. BASIC DATASET INFORMATION")
print("-" * 30)
print(f"Columns: {list(df.columns)}")
print(f"\nData Types:")
print(df.dtypes)
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# 3. First few rows
print("\n3. FIRST 5 ROWS")
print("-" * 30)
print(df.head())

# 4. Missing Values Analysis
print("\n4. MISSING VALUES ANALYSIS")
print("-" * 30)
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])
if missing_df['Missing Count'].sum() == 0:
    print("✓ No missing values found in the dataset!")

# 5. Summary Statistics
print("\n5. NUMERICAL FEATURES SUMMARY")
print("-" * 30)
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(df[numerical_cols].describe())

print("\n6. CATEGORICAL FEATURES SUMMARY")
print("-" * 30)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col.upper()}:")
    print(df[col].value_counts())
    print(f"Unique values: {df[col].nunique()}")

# 7. Distribution Plots for Numerical Features
print("\n7. CREATING VISUALIZATIONS...")
print("-" * 30)

# Create subplots for numerical distributions
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(numerical_cols[:4]):  # Plot first 4 numerical columns
        row = i // 2
        col_idx = i % 2
        sns.histplot(data=df, x=col, kde=True, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Numerical distributions plot saved as 'numerical_distributions.png'")

# 8. Categorical Features Analysis
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of Categorical Features', fontsize=16, fontweight='bold')
    
    for i, col in enumerate(categorical_cols[:4]):  # Plot first 4 categorical columns
        row = i // 2
        col_idx = i % 2
        value_counts = df[col].value_counts()
        axes[row, col_idx].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        axes[row, col_idx].set_title(f'Distribution of {col}', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Categorical distributions plot saved as 'categorical_distributions.png'")

# 9. Correlation Analysis
print("\n8. CORRELATION ANALYSIS")
print("-" * 30)
numerical_df = df.select_dtypes(include=[np.number])
if len(numerical_df.columns) > 1:
    correlation_matrix = numerical_df.corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")

# 10. Target Variable Analysis (if exists)
print("\n9. TARGET VARIABLE ANALYSIS")
print("-" * 30)
target_candidates = ['loan_approved', 'risk_level', 'approved']
target_found = False

for target in target_candidates:
    if target in df.columns:
        print(f"Target variable found: {target}")
        print(f"Distribution:")
        print(df[target].value_counts())
        print(f"Percentage distribution:")
        print((df[target].value_counts() / len(df) * 100).round(2))
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=target)
        plt.title(f'Distribution of {target}', fontsize=16, fontweight='bold')
        plt.xlabel(target)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{target}_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ {target} distribution plot saved as '{target}_distribution.png'")
        target_found = True
        break

if not target_found:
    print("No standard target variable found in the dataset.")

# 11. Business Insights
print("\n10. BUSINESS INSIGHTS")
print("-" * 30)

# Revenue Analysis
if 'monthly_revenue' in df.columns:
    print(f"Revenue Analysis:")
    print(f"  - Average Monthly Revenue: ₹{df['monthly_revenue'].mean():,.2f}")
    print(f"  - Median Monthly Revenue: ₹{df['monthly_revenue'].median():,.2f}")
    print(f"  - Min Revenue: ₹{df['monthly_revenue'].min():,.2f}")
    print(f"  - Max Revenue: ₹{df['monthly_revenue'].max():,.2f}")

# Credit Score Analysis
if 'credit_score' in df.columns:
    print(f"\nCredit Score Analysis:")
    print(f"  - Average Credit Score: {df['credit_score'].mean():.2f}")
    print(f"  - Median Credit Score: {df['credit_score'].median():.2f}")
    print(f"  - Min Credit Score: {df['credit_score'].min()}")
    print(f"  - Max Credit Score: {df['credit_score'].max()}")

# Business Age Analysis
if 'business_age' in df.columns:
    print(f"\nBusiness Age Analysis:")
    print(f"  - Average Business Age: {df['business_age'].mean():.2f} years")
    print(f"  - Median Business Age: {df['business_age'].median():.2f} years")
    print(f"  - Min Business Age: {df['business_age'].min()} years")
    print(f"  - Max Business Age: {df['business_age'].max()} years")

# GST Compliance Analysis
if 'gst_compliance' in df.columns:
    print(f"\nGST Compliance Analysis:")
    gst_counts = df['gst_compliance'].value_counts()
    print(f"  - GST Compliant: {gst_counts.get('Yes', 0)} ({gst_counts.get('Yes', 0)/len(df)*100:.1f}%)")
    print(f"  - Not GST Compliant: {gst_counts.get('No', 0)} ({gst_counts.get('No', 0)/len(df)*100:.1f}%)")

# Sector Analysis
if 'business_type' in df.columns:
    print(f"\nBusiness Sector Analysis:")
    sector_counts = df['business_type'].value_counts()
    print("  Top 5 sectors:")
    for i, (sector, count) in enumerate(sector_counts.head().items(), 1):
        print(f"    {i}. {sector}: {count} ({count/len(df)*100:.1f}%)")

# 12. Data Quality Assessment
print("\n11. DATA QUALITY ASSESSMENT")
print("-" * 30)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"✓ Duplicate rows: {duplicates}")

# Check for outliers in numerical columns
print(f"\nOutlier Analysis (using IQR method):")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# 13. Summary Report
print("\n12. SUMMARY REPORT")
print("-" * 30)
print(f"✓ Dataset contains {len(df)} SME loan applications")
print(f"✓ {len(df.columns)} features analyzed")
print(f"✓ {len(numerical_cols)} numerical features")
print(f"✓ {len(categorical_cols)} categorical features")
print(f"✓ Data quality: {'Good' if duplicates == 0 and missing_df['Missing Count'].sum() == 0 else 'Needs attention'}")
print(f"✓ Visualizations saved as PNG files")

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60) 