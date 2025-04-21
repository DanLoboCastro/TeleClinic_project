import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from faker import Faker

# Set seed for reproducibility
np.random.seed(42)

# Initialize the Faker generator for fake names
fake = Faker()

# Function to generate original medical insurance data
def generate_insurance_data(n_samples=1000):
    data = {}

    # ID column (will be repeated later for multiple claims per patient)
    data['id'] = [f'P{str(i+1).zfill(5)}' for i in range(n_samples)]

    # Generate fake names for patients
    data['patient_name'] = [fake.name() for _ in range(n_samples)]

    # Basic demographics
    data['age'] = np.random.randint(18, 66, size=n_samples)
    data['sex'] = np.random.choice(['male', 'female'], size=n_samples)
    data['bmi'] = np.round(np.random.normal(30, 6, size=n_samples), 1)
    data['bmi'] = np.clip(data['bmi'], 15, 55)
    data['smoker'] = np.random.choice(['yes', 'no'], size=n_samples, p=[0.2, 0.8])
    data['region'] = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], size=n_samples)

    # Medical history
    data['has_diabetes'] = np.random.choice(['yes', 'no'], size=n_samples, p=[0.15, 0.85])
    data['has_hypertension'] = np.random.choice(['yes', 'no'], size=n_samples, p=[0.2, 0.8])
    data['has_heart_disease'] = np.random.choice(['yes', 'no'], size=n_samples, p=[0.1, 0.9])
    data['physical_activity_level'] = np.random.choice(['low', 'moderate', 'high'], size=n_samples, p=[0.3, 0.5, 0.2])
    data['cholesterol_level'] = np.round(np.random.normal(200, 30, size=n_samples))
    data['cholesterol_level'] = np.clip(data['cholesterol_level'], 120, 300)

    # Charges calculation
    base_charge = 2000
    smoker_charge = 20000
    bmi_factor = 300
    age_factor = 250
    diabetes_penalty = 5000
    hypertension_penalty = 4000
    heart_disease_penalty = 10000
    activity_discount = {'low': 0, 'moderate': -1000, 'high': -2000}
    cholesterol_penalty = lambda chol: max(0, chol - 200) * 20

    charges = (
        base_charge
        + data['age'] * age_factor
        + data['bmi'] * bmi_factor
        + np.where(np.array(data['smoker']) == 'yes', smoker_charge, 0)
        + np.where(np.array(data['has_diabetes']) == 'yes', diabetes_penalty, 0)
        + np.where(np.array(data['has_hypertension']) == 'yes', hypertension_penalty, 0)
        + np.where(np.array(data['has_heart_disease']) == 'yes', heart_disease_penalty, 0)
        + [activity_discount[pa] for pa in data['physical_activity_level']]
        + [cholesterol_penalty(chol) for chol in data['cholesterol_level']]
        + np.random.normal(0, 1000, size=n_samples)  # noise
    )

    data['charges'] = np.round(charges, 2)

    return pd.DataFrame(data)

# Function to generate multiple claims data for each patient
def generate_multiple_claims(df, claims_per_patient=3):
    all_claims = []

    for index, row in df.iterrows():
        patient_id = row['id']
        for i in range(claims_per_patient):
            claim_id = f'{patient_id}_C{i+1}'
            claim_data = row.copy()
            claim_data['claim_id'] = claim_id

            # Adjust claims data
            claim_data['claim_status'] = np.random.choice(['Pending', 'Approved', 'Denied'], p=[0.3, 0.6, 0.1])
            claim_data['submission_date'] = datetime.today() - timedelta(days=np.random.randint(1, 30))

            # Adjust the claim amount and reimbursement based on different claim conditions
            claim_data['total_claims'] = np.random.normal(5000, 1000) + row['charges']
            claim_data['reimbursed_amount'] = claim_data['total_claims'] * 0.8  # Simulated reimbursement logic
            claim_data['out_of_pocket'] = claim_data['total_claims'] - claim_data['reimbursed_amount']

            all_claims.append(claim_data)

    # Create DataFrame from all the claims
    claims_df = pd.DataFrame(all_claims)
    return claims_df

# Function to enhance claims data with more details
def enhance_claims_data(df):
    today = datetime.today()

    # Generate the 'documentation_uploaded' column first
    df['documentation_uploaded'] = np.random.choice(['Yes', 'No'], size=len(df), p=[0.9, 0.1])

    # Criteria met logic
    df['reimbursement_criteria_met'] = np.where(
        (df['smoker'] == 'no') & (df['documentation_uploaded'] == 'Yes'),
        'Yes',
        'No'
    )
    
    # Dates
    df['approval_date'] = [
        date + timedelta(days=np.random.randint(3, 15)) if status == 'Approved' else pd.NaT
        for date, status in zip(df['submission_date'], df['claim_status'])
    ]
    df['expected_reimbursement_date'] = [
        date + timedelta(days=30) if pd.notnull(date) else pd.NaT
        for date in df['approval_date']
    ]

    # Missing docs and denial reasons
    df['missing_documents'] = np.where(
        (df['reimbursement_criteria_met'] == 'No') & (df['claim_status'] != 'Approved'),
        np.random.choice(['ID Proof', 'Lab Report', 'Referral Note', 'None'], size=len(df)),
        'None'
    )
    df['reason_for_denial'] = np.where(
        df['claim_status'] == 'Denied',
        np.random.choice(['Out-of-Network Provider', 'Missing Documents', 'Non-covered Service'], size=len(df)),
        ''
    )

    # Physician side fields
    df['procedure_code'] = np.random.choice(['99213', '99214', '99203', '99385', '99406'], size=len(df))
    df['visit_type'] = np.random.choice(['Routine', 'Specialist', 'Emergency'], size=len(df), p=[0.6, 0.3, 0.1])
    df['prior_authorization_required'] = np.random.choice(['Yes', 'No'], size=len(df), p=[0.2, 0.8])
    df['network_status'] = np.random.choice(['In-Network', 'Out-of-Network'], size=len(df), p=[0.85, 0.15])

    return df

# Generate and merge data
df_insurance = generate_insurance_data(1000)

# Add multiple claims for each patient
df_claims = generate_multiple_claims(df_insurance)

# Enhance the claims data with extra details
df_final = enhance_claims_data(df_claims)

# Now, filter data for one patient (for example, patient with ID 'P00001')
patient_id = 'P00001'
patient_data = df_final[df_final['id'] == patient_id]

# Save to a separate CSV file
patient_data.to_csv(f'{patient_id}_claims.csv', index=False)

# Show a preview of the patient-specific data
print(patient_data.head(3))

