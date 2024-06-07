import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

# Load the dataset
st.title("Hepatitis Awareness and Prediction")
st.image("hepatitis.jpg")

st.write("Upload your hepatitis dataset as a CSV file.")
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    # Basic data statistics
    st.write("### Data Summary")
    st.write(df.describe())
    
    # Handling missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    # Data preprocessing
    st.write("### Data Preprocessing")
    
    # Feature Engineering - Selecting top features based on correlation
    corr = df.corr()
    top_features = corr['target'].abs().sort_values(ascending=False).index[1:6]
    st.write("### Top Features", top_features)
    
    # Data Visualization
    st.write("### Exploratory Data Analysis")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, feature in enumerate(top_features):
        sns.histplot(df[feature], kde=True, ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f"Distribution of {feature}")
    st.pyplot(fig)
    
    # Train-test split
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model Training Using 3 for now
    st.write("### Model Training and Evaluation")
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC()
    }
    
    # Evaluating models and then comparing their metrics

    model_performance = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        model_performance[name] = [accuracy, precision, recall, f1]
    
    performance_df = pd.DataFrame(model_performance, index=["Accuracy", "Precision", "Recall", "F1 Score"])
    st.write(performance_df)
    st.write("### Model Performance Comparison")
    st.bar_chart(performance_df.T)
    
    # Taking input from user and then using that input for prediction
    st.write("### Prediction Inputs")
    user_inputs = {}
    for feature in top_features:
        if df[feature].dtype == 'float64' or df[feature].dtype == 'int64':
            user_inputs[feature] = st.slider(f"Enter value for {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        else:
            unique_vals = df[feature].unique()
            user_inputs[feature] = st.selectbox(f"Select value for {feature}", unique_vals)
    
    if st.button("Predict"):
        # Prepare full input vector with all features
        full_input = df.drop(columns=['target']).mean().values  # default to mean values
        for feature in top_features:
            full_input[df.columns.get_loc(feature) - 1] = user_inputs[feature]  # -1 because target is dropped
        
        # Scale the input (scikit learn do not accept 1D)
        full_input_scaled = scaler.transform(full_input.reshape(1, -1))
        
        # Predict with the chosen model
        prediction = models["Random Forest"].predict(full_input_scaled)
        st.write(f"Prediction (1 = DIE, 2 = LIVE): {prediction[0]}")
