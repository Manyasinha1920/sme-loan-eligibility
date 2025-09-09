from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import re

app = Flask(__name__)
CORS(app)

class SMELoanPredictor:
    def __init__(self):
        self.approval_model = None
        self.loan_amount_model = None
        self.interest_rate_model = None
        self.risk_model = None
        self.scalers = {}
        self.encoders = {}
        self.is_trained = False
        
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Create a copy to avoid modifying original data
        data = df.copy()

        # Rename columns to match expected names in code
        data = data.rename(columns={
            'business_type': 'sector',
            'gst_compliance': 'gst_compliant',
            'loan_approved': 'approved',
            'pan_card': 'pan',
            'gst_number': 'gst_no',
            'max_loan_amount': 'max_loan'
        })

        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Convert boolean columns
        data['gst_compliant'] = data['gst_compliant'].astype(bool)
        data['approved'] = data['approved'].astype(bool)

        # Encode categorical variables
        le_sector = LabelEncoder()
        le_risk = LabelEncoder()

        data['sector_encoded'] = le_sector.fit_transform(data['sector'])
        data['risk_encoded'] = le_risk.fit_transform(data['risk_level'])

        self.encoders['sector'] = le_sector
        self.encoders['risk'] = le_risk

        return data
    
    def train_models(self, csv_path):
        """Train all ML models"""
        try:
            # Load data
            df = pd.read_csv(csv_path)
            print(f"Loaded dataset with {len(df)} entries")

            # Check if we have sufficient data
            if len(df) < 50:
                print(f"WARNING: Only {len(df)} entries found. Minimum 50 recommended for reliable predictions.")

            data = self.preprocess_data(df)
            print(f"Training models with {len(data)} entries...")

            # Features for prediction
            feature_cols = ['business_age', 'monthly_revenue', 'credit_score',
                          'bank_balance', 'gst_compliant', 'sector_encoded']
            # If 'employees' column exists, use it, else default to 10
            if 'employees' in data.columns:
                feature_cols.append('employees')
            else:
                data['employees'] = 10
                feature_cols.append('employees')

            X = data[feature_cols]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['features'] = scaler

            # Train approval model
            y_approval = data['approved']
            self.approval_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.approval_model.fit(X_scaled, y_approval)

            # Train loan amount model (only on approved loans)
            approved_data = data[data['approved'] == True]
            if len(approved_data) > 0:
                X_approved = approved_data[feature_cols]
                X_approved_scaled = scaler.transform(X_approved)
                y_loan = approved_data['max_loan']

                self.loan_amount_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.loan_amount_model.fit(X_approved_scaled, y_loan)

                # Train interest rate model
                y_interest = approved_data['interest_rate']
                self.interest_rate_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.interest_rate_model.fit(X_approved_scaled, y_interest)

            # Train risk model
            y_risk = data['risk_encoded']
            self.risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.risk_model.fit(X_scaled, y_risk)

            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def predict(self, input_data):
        """Make predictions for new data"""
        if not self.is_trained:
            return None
            
        try:
            # Prepare input data
            features = np.array([[
                input_data['business_age'],
                input_data['monthly_revenue'],
                input_data['credit_score'],
                input_data['annual_revenue'],
                input_data['gst_compliant'],
                input_data['employees'],
                input_data['sector_encoded']
            ]])
            
            # Scale features
            features_scaled = self.scalers['features'].transform(features)
            
            # Predict approval probability
            approval_prob = self.approval_model.predict_proba(features_scaled)[0][1]
            approval_pred = self.approval_model.predict(features_scaled)[0]
            
            # Predict risk level
            risk_pred = self.risk_model.predict(features_scaled)[0]
            risk_level = self.encoders['risk'].inverse_transform([risk_pred])[0]
            
            # Predict loan amount and interest rate if likely to be approved
            max_loan = 0
            interest_rate = 0
            
            if approval_prob > 0.5 and self.loan_amount_model and self.interest_rate_model:
                max_loan = max(0, min(10000000, self.loan_amount_model.predict(features_scaled)[0]))
                interest_rate = max(8, min(20, self.interest_rate_model.predict(features_scaled)[0]))
            
            return {
                'approval_probability': round(approval_prob * 100, 2),
                'likely_approved': bool(approval_pred),
                'risk_level': risk_level,
                'max_loan_amount': int(max_loan),
                'interest_rate': round(interest_rate, 2)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

# Initialize predictor
predictor = SMELoanPredictor()

# Load and train models on startup
try:
    if os.path.exists('improved_sme_loan_data2004.csv'):
        predictor.train_models('improved_sme_loan_data2004.csv')
        print("Models trained successfully!")
    else:
        print("CSV file not found. Please ensure improved_sme_loan_data2004.csv is in the project directory.")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_loan():
    try:
        data = request.get_json()
        # Convert numeric fields to int
        data['credit_score'] = int(data['credit_score'])
        data['monthly_revenue'] = int(data['monthly_revenue'])
        data['business_age'] = int(data['business_age'])
        data['bank_balance'] = int(data.get('bank_balance', 0))
        # Convert gst_compliant from 'Yes'/'No' to boolean
        if isinstance(data['gst_compliant'], str):
            data['gst_compliant'] = data['gst_compliant'].strip().lower() == 'yes'
        
        # Validate required fields
        required_fields = ['monthly_revenue', 'credit_score', 'business_age', 
                          'gst_compliant', 'sector', 'pan', 'gst_no']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Basic validation
        if not (300 <= data['credit_score'] <= 900):
            return jsonify({'error': 'Credit score must be between 300 and 900'}), 400
            
        if data['monthly_revenue'] <= 0:
            return jsonify({'error': 'Monthly revenue must be positive'}), 400
            
        if data['business_age'] < 0:
            return jsonify({'error': 'Business age cannot be negative'}), 400
        
        # Validate PAN format (basic)
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', data['pan']):
            return jsonify({'error': 'Invalid PAN format'}), 400
        
        # Validate GST format (basic)
        if not re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$', data['gst_no']):
            return jsonify({'error': 'Invalid GST format'}), 400
        
        # Prepare data for prediction
        sector_mapping = {
            'Construction': 0,
            'Trading': 1,
            'Textile': 2,
            'Manufacturing': 3,
            'Services': 4,
            'Technology': 5,
            'Healthcare': 6,
            'Education': 7,
            'Food': 8,
            'Retail': 9
        }
        
        input_data = {
            'business_age': data['business_age'],
            'monthly_revenue': data['monthly_revenue'],
            'credit_score': data['credit_score'],
            'annual_revenue': data['monthly_revenue'] * 12,  # Calculate annual revenue
            'gst_compliant': data['gst_compliant'],
            'employees': data.get('employees', 10),  # Default to 10 if not provided
            'sector_encoded': sector_mapping.get(data['sector'], 0)
        }
        
        # Make prediction
        if predictor.is_trained:
            prediction = predictor.predict(input_data)
            if prediction:
                return jsonify({
                    'success': True,
                    'prediction': prediction
                })
            else:
                return jsonify({'error': 'Error making prediction'}), 500
        else:
            return jsonify({'error': 'Models not trained yet'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)