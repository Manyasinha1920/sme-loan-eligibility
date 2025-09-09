import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import string

class SMEDataGenerator:
    def __init__(self):
        self.sectors = [
            'Construction', 'Trading', 'Textile', 'Manufacturing', 'Services',
            'Technology', 'Healthcare', 'Food & Beverage', 'Retail', 'Transport',
            'Agriculture', 'Education', 'Finance', 'Real Estate', 'Tourism'
        ]
        
        self.pan_prefixes = ['ABCDE', 'FGHIJ', 'KLMNO', 'PQRST', 'UVWXY']
        self.gst_states = ['27', '29', '32', '33', '36', '24', '23', '19', '09', '06']
        
    def generate_pan(self):
        """Generate a realistic PAN number format"""
        letters = ''.join(random.choices(string.ascii_uppercase, k=5))
        digits = ''.join(random.choices(string.digits, k=4))
        last_letter = random.choice(string.ascii_uppercase)
        return f"{letters}{digits}{last_letter}"
    
    def generate_gst(self):
        """Generate a realistic GST number format"""
        state_code = random.choice(self.gst_states)
        pan_part = self.generate_pan()[:10]
        entity_code = random.choice(['1', '2', '3', '4'])
        check_digit = random.choice(string.ascii_uppercase + string.digits)
        return f"{state_code}{pan_part}{entity_code}{check_digit}"
    
    def calculate_risk_and_approval(self, credit_score, business_age, monthly_revenue, gst_compliant):
        """Calculate risk level and approval based on business metrics"""
        risk_score = 0
        
        # Credit score impact (40% weight)
        if credit_score >= 750:
            risk_score += 40
        elif credit_score >= 650:
            risk_score += 25
        elif credit_score >= 550:
            risk_score += 10
        else:
            risk_score += 0
            
        # Business age impact (20% weight)
        if business_age >= 5:
            risk_score += 20
        elif business_age >= 2:
            risk_score += 15
        elif business_age >= 1:
            risk_score += 10
        else:
            risk_score += 5
            
        # Monthly revenue impact (25% weight)
        if monthly_revenue >= 5000000:  # 50L+
            risk_score += 25
        elif monthly_revenue >= 2000000:  # 20L+
            risk_score += 20
        elif monthly_revenue >= 1000000:  # 10L+
            risk_score += 15
        elif monthly_revenue >= 500000:   # 5L+
            risk_score += 10
        else:
            risk_score += 5
            
        # GST compliance impact (15% weight)
        if gst_compliant:
            risk_score += 15
        else:
            risk_score += 5
            
        # Determine risk level
        if risk_score >= 75:
            risk_level = 'Low'
            approval_prob = 0.9
        elif risk_score >= 50:
            risk_level = 'Medium'
            approval_prob = 0.7
        else:
            risk_level = 'High'
            approval_prob = 0.3
            
        # Add some randomness
        approval_prob += random.uniform(-0.1, 0.1)
        approved = random.random() < max(0.1, min(0.95, approval_prob))
        
        return risk_level, approved
    
    def calculate_max_loan(self, monthly_revenue, credit_score, business_age, risk_level, approved):
        """Calculate maximum loan amount based on business metrics"""
        if not approved:
            return random.randint(100000, 1000000)  # Lower amounts for rejected loans
        
        # Base loan calculation (typically 6-12x monthly revenue)
        base_multiplier = random.uniform(6, 12)
        
        # Adjust based on credit score
        if credit_score >= 750:
            base_multiplier *= 1.2
        elif credit_score >= 650:
            base_multiplier *= 1.0
        elif credit_score >= 550:
            base_multiplier *= 0.8
        else:
            base_multiplier *= 0.6
            
        # Adjust based on business age
        if business_age >= 5:
            base_multiplier *= 1.1
        elif business_age >= 2:
            base_multiplier *= 1.0
        else:
            base_multiplier *= 0.9
            
        # Adjust based on risk level
        if risk_level == 'Low':
            base_multiplier *= 1.1
        elif risk_level == 'Medium':
            base_multiplier *= 1.0
        else:
            base_multiplier *= 0.8
            
        max_loan = int(monthly_revenue * base_multiplier)
        
        # Cap at 1 crore and ensure minimum
        max_loan = min(max_loan, 10000000)  # 1 crore max
        max_loan = max(max_loan, 100000)    # 1 lakh min
        
        return max_loan
    
    def calculate_interest_rate(self, credit_score, risk_level, business_age):
        """Calculate interest rate based on risk factors"""
        base_rate = 12.0  # Base rate
        
        # Credit score adjustment
        if credit_score >= 750:
            base_rate -= 2.0
        elif credit_score >= 650:
            base_rate -= 1.0
        elif credit_score >= 550:
            base_rate += 1.0
        else:
            base_rate += 3.0
            
        # Risk level adjustment
        if risk_level == 'Low':
            base_rate -= 1.0
        elif risk_level == 'High':
            base_rate += 2.0
            
        # Business age adjustment
        if business_age >= 5:
            base_rate -= 0.5
        elif business_age < 1:
            base_rate += 1.0
            
        # Add some randomness
        base_rate += random.uniform(-0.5, 0.5)
        
        # Ensure reasonable bounds
        return round(max(8.0, min(18.0, base_rate)), 2)
    
    def generate_sme_data(self, num_records=1000):
        """Generate synthetic SME loan data"""
        data = []
        
        for i in range(num_records):
            sme_id = f"SME{str(i+1).zfill(4)}"
            sector = random.choice(self.sectors)
            business_age = random.randint(0, 20)
            
            # Generate monthly revenue with sector-based variation
            if sector in ['Technology', 'Finance', 'Healthcare']:
                monthly_revenue = random.randint(1000000, 8000000)  # Higher for tech/finance
            elif sector in ['Construction', 'Manufacturing', 'Real Estate']:
                monthly_revenue = random.randint(500000, 6000000)   # Medium-high for construction
            elif sector in ['Trading', 'Retail', 'Services']:
                monthly_revenue = random.randint(200000, 4000000)   # Medium for trading
            else:
                monthly_revenue = random.randint(100000, 3000000)   # Lower for others
            
            # Annual revenue (with some variation from monthly * 12)
            annual_revenue = int(monthly_revenue * 12 * random.uniform(0.9, 1.1))
            
            # Credit score with realistic distribution
            credit_score = int(np.random.beta(2, 2) * 400 + 300)  # Bell curve between 300-700
            credit_score = max(300, min(850, credit_score))
            
            # GST compliance (higher probability for larger businesses)
            gst_prob = 0.4 + (monthly_revenue / 10000000) * 0.4  # 40-80% based on revenue
            gst_compliant = random.random() < min(0.85, gst_prob)
            
            # Generate PAN and GST numbers
            pan = self.generate_pan()
            gst_no = self.generate_gst() if gst_compliant else ""
            
            # Calculate risk and approval
            risk_level, approved = self.calculate_risk_and_approval(
                credit_score, business_age, monthly_revenue, gst_compliant
            )
            
            # Calculate employees based on revenue
            if monthly_revenue >= 5000000:
                employees = random.randint(50, 200)
            elif monthly_revenue >= 2000000:
                employees = random.randint(20, 100)
            elif monthly_revenue >= 1000000:
                employees = random.randint(10, 50)
            else:
                employees = random.randint(1, 25)
            
            # Calculate max loan and interest rate
            max_loan = self.calculate_max_loan(
                monthly_revenue, credit_score, business_age, risk_level, approved
            )
            interest_rate = self.calculate_interest_rate(credit_score, risk_level, business_age)
            
            record = {
                'sme_id': sme_id,
                'sector': sector,
                'business_age': business_age,
                'monthly_revenue': monthly_revenue,
                'credit_score': credit_score,
                'annual_revenue': annual_revenue,
                'gst_compliant': gst_compliant,
                'pan': pan,
                'gst_no': gst_no,
                'risk_level': risk_level,
                'employees': employees,
                'max_loan': max_loan,
                'interest_rate': interest_rate,
                'approved': approved
            }
            
            data.append(record)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} records...")
        
        return pd.DataFrame(data)

def main():
    """Generate and save SME loan dataset"""
    print("Starting SME Loan Data Generation...")
    
    generator = SMEDataGenerator()
    
    # Generate dataset
    num_records = int(input("Enter number of records to generate (default 1000): ") or 1000)
    df = generator.generate_sme_data(num_records)
    
    # Display basic statistics
    print(f"\nDataset generated with {len(df)} records")
    print(f"Approval rate: {df['approved'].mean():.2%}")
    print(f"Risk distribution:")
    print(df['risk_level'].value_counts())
    print(f"\nSector distribution:")
    print(df['sector'].value_counts())
    
    # Save to CSV
    filename = f"sme_loan_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nDataset saved as: {filename}")
    
    # Display sample records
    print("\nSample records:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    dataset = main()