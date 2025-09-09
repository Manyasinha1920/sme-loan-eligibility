import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('improved_sme_loan_data2004.csv')

# Example: Adjust these columns as per your dataset
features = ['monthly_revenue', 'credit_score', 'bank_balance', 'business_age', 'gst_compliance']
target = 'risk'  # Should be 'low', 'medium', 'high' in your dataset

# Preprocessing (example, adjust as needed)
df['gst_compliance'] = df['gst_compliance'].map({'Yes': 1, 'No': 0})

X = df[features]
y = df[target]

y = y.map({'low': 0, 'medium': 1, 'high': 2})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'sme_loan_model.pkl') 