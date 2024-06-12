import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess data
@st.cache_data
def load_data():
    # Replace the path with the correct file path
    data = pd.read_csv('cc_approvals.data', header=None)
    data.columns = [
        'Gender', 'Age', 'Debt', 'Marital_Status', 'Bank_Customer', 'Education_Level', 
        'Ethnicity', 'Years_Employed', 'Prior_Default', 'Employed', 'Credit_Score', 
        'Drivers_License', 'Citizen', 'Approval_Status'
    ]
    data.replace('?', np.nan, inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)
    data['Age'] = data['Age'].astype(float)
    data['Debt'] = data['Debt'].astype(float)
    data['Years_Employed'] = data['Years_Employed'].astype(float)
    data['Credit_Score'] = data['Credit_Score'].astype(int)
    data['Approval_Status'] = data['Approval_Status'].apply(lambda x: 1 if x == '+' else 0)

    binary_columns = ['Prior_Default', 'Employed', 'Drivers_License']
    for col in binary_columns:
        data[col] = data[col].apply(lambda x: 1 if x == 't' else 0)

    data = pd.get_dummies(data, columns=['Gender', 'Marital_Status', 'Bank_Customer', 'Education_Level', 'Ethnicity'], drop_first=True)

    return data

data = load_data()

# Split data into features and target
X = data.drop('Approval_Status', axis=1)
y = data['Approval_Status']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the numerical features
scaler = StandardScaler()
X_train[['Age', 'Debt', 'Years_Employed', 'Credit_Score']] = scaler.fit_transform(X_train[['Age', 'Debt', 'Years_Employed', 'Credit_Score']])
X_test[['Age', 'Debt', 'Years_Employed', 'Credit_Score']] = scaler.transform(X_test[['Age', 'Debt', 'Years_Employed', 'Credit_Score']])

# Train the model
model = LogisticRegression(C=100, penalty='l2', solver='liblinear')
model.fit(X_train, y_train)

# Streamlit UI
st.title("Credit Card Approval Predictor")

st.sidebar.header("Applicant's Details")
def user_input_features():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    debt = st.sidebar.number_input("Debt", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    years_employed = st.sidebar.number_input("Years Employed", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    credit_score = st.sidebar.number_input("Credit Score", min_value=0, max_value=100000, value=1000)
    prior_default = st.sidebar.selectbox("Prior Default", options=['No', 'Yes'])
    employed = st.sidebar.selectbox("Employed", options=['No', 'Yes'])
    drivers_license = st.sidebar.selectbox("Drivers License", options=['No', 'Yes'])
    gender = st.sidebar.selectbox("Gender", options=['a', 'b'])
    marital_status = st.sidebar.selectbox("Marital Status", options=['u', 'y'])
    bank_customer = st.sidebar.selectbox("Bank Customer", options=['g', 'p'])
    education_level = st.sidebar.selectbox("Education Level", options=['w', 'q', 'm', 'r', 'cc', 'k', 'x'])
    ethnicity = st.sidebar.selectbox("Ethnicity", options=['v', 'h', 'bb', 'j', 'ff', 'dd', 'o', 'z', 'n'])

    data = {
        'Age': age,
        'Debt': debt,
        'Years_Employed': years_employed,
        'Credit_Score': credit_score,
        'Prior_Default': 1 if prior_default == 'Yes' else 0,
        'Employed': 1 if employed == 'Yes' else 0,
        'Drivers_License': 1 if drivers_license == 'Yes' else 0,
        'Gender_b': 1 if gender == 'b' else 0,
        'Marital_Status_y': 1 if marital_status == 'y' else 0,
        'Bank_Customer_p': 1 if bank_customer == 'p' else 0,
        'Education_Level_q': 1 if education_level == 'q' else 0,
        'Education_Level_m': 1 if education_level == 'm' else 0,
        'Education_Level_r': 1 if education_level == 'r' else 0,
        'Education_Level_cc': 1 if education_level == 'cc' else 0,
        'Education_Level_k': 1 if education_level == 'k' else 0,
        'Education_Level_x': 1 if education_level == 'x' else 0,
        'Ethnicity_h': 1 if ethnicity == 'h' else 0,
        'Ethnicity_bb': 1 if ethnicity == 'bb' else 0,
        'Ethnicity_j': 1 if ethnicity == 'j' else 0,
        'Ethnicity_ff': 1 if ethnicity == 'ff' else 0,
        'Ethnicity_dd': 1 if ethnicity == 'dd' else 0,
        'Ethnicity_o': 1 if ethnicity == 'o' else 0,
        'Ethnicity_z': 1 if ethnicity == 'z' else 0,
        'Ethnicity_n': 1 if ethnicity == 'n' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Standardize the input data
input_df[['Age', 'Debt', 'Years_Employed', 'Credit_Score']] = scaler.transform(input_df[['Age', 'Debt', 'Years_Employed', 'Credit_Score']])

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
credit_status = np.array(['Rejected', 'Approved'])
st.write(credit_status[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)

