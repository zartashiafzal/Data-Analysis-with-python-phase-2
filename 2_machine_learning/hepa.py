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


st.sidebar.title("Navigation")
page= st.sidebar.radio("Go to", ["Data Analysis and Prediction", "Hepatitis Awareness", "Contact us"])

if page == "Data Analysis and Prediction":

    # Load the dataset
    st.title("Hepatitis Awareness and Prediction")
    st.image("hepatitis.jpg")

    st.write("Upload your hepatitis dataset as a CSV file.")
    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, na_values="?")
        st.write("### Data Preview", df.head())
        
        # Basic data statistics
        st.write("### Data Summary")
        st.write(df.describe())
        
        # Handling missing values
        st.write("### Missing Values")
        st.write(df.isnull().sum())

        # Data preprocessing
        st.write("### Data Preprocessing")

        # Handling missing values for categorical and numerical variables

        # Convert appropriate columns to numeric, errors='coerce' will handle any improper conversion
        numerical_cols = ['age', 'bili', 'alk', 'sgot', 'albu', 'protime']
        df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')

        # Identify the actual categorical columns based on your metadata
        categorical_cols = ['gender', 'steroid', 'antivirals', 'fatigue', 'malaise', 
                            'anorexia', 'liverBig', 'liverFirm', 'spleen', 'spiders',
                            'ascites', 'varices', 'histology']

        # Convert these categorical columns to type 'category'
        df[categorical_cols] = df[categorical_cols].astype('category')

        # Handling missing values for numerical and categorical variables

        # Fill missing numerical values with the median of each column
        for col in numerical_cols:
            if df[col].isnull().any():  # Only fill if there are missing values
                df[col] = df[col].fillna(df[col].median())

        # Fill missing categorical values with the mode of each column
        for col in categorical_cols:
            if df[col].isnull().any():  # Only fill if there are missing values
                mode_value = df[col].mode()
                if not mode_value.empty:  # Check if mode exists
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    # If mode is not found (e.g., all values are NaN), fill with a placeholder
                    df[col] = df[col].fillna('Unknown')

        # After handling missing values, confirm the missing values are handled
        st.write("### Missing Values After Treatment")
        st.write(df.isnull().sum())

        
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

        ### Plotly
    

        # Ensure data types are correct
        df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
        df[categorical_cols] = df[categorical_cols].astype('category')

        # Animated Plotly Bar Plot
        # Let's visualize the count of patients by 'steroid' across 'gender' with animation
        st.write("### Animated Plotly Bar Graph - Steroid Usage Across Gender")

        # Remove rows with missing 'gender' or 'steroid' for this visualization
        df_plot = df.dropna(subset=['gender', 'steroid'])

        # Create an animated bar plot
        fig = px.bar(df_plot, x='gender', y='steroid', color='steroid', animation_frame='gender',
                    title='Steroid Usage Across Gender',
                    labels={'steroid': 'Steroid Usage (1 = Yes, 2 = No)', 'gender': 'Gender (1 = Male, 2 = Female)'},
                    category_orders={'gender': ['1', '2'], 'steroid': ['1', '2']},  # Ensure consistent ordering
                    barmode='group')

        # Slow down the animation
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000  # 1000ms per frame

        fig.update_layout(transition=dict(duration=1000), showlegend=True)

        # Show the animated plot in Streamlit
        st.plotly_chart(fig)



    ### ______________________________________    
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


elif page == "Hepatitis Awareness":
    st.title("Hepatitis Awareness")
    st.write(
"""
    ### What is Hepatitis?
    Hepatitis refers to an inflammatory condition of the liver. It’s commonly caused by a viral infection, but there are other possible causes of hepatitis. These include autoimmune hepatitis and hepatitis that occurs as a secondary result of medications, drugs, toxins, and alcohol.

    Autoimmune hepatitis is a disease that occurs when your body makes antibodies against your liver tissue.

    Your liver is located in the right upper area of your abdomen. It performs many critical functions that affect metabolism throughout your body, including:

    - Bile production, which is essential to digestion
    - Filtering of toxins from your body
    - Excretion of bilirubin (a product of broken-down red blood cells), cholesterol, hormones, and drugs
    - Metabolism of carbohydrates, fats, and proteins
    - Activation of enzymes, which are specialized proteins essential to body functions
    - Storage of glycogen (a form of sugar), minerals, and vitamins (A, D, E, and K)
    - Synthesis of blood proteins, such as albumin
    - Synthesis of clotting factors

    When the liver is inflamed or damaged, its functions can be affected.

    ### Types of Hepatitis
    - **Hepatitis A:** Hepatitis A is caused by consuming contaminated food or water. This type is often acute and resolves without treatment.
    - **Hepatitis B:** Hepatitis B is spread through contact with infectious body fluids. It can be both acute and chronic.
    - **Hepatitis C:** Hepatitis C is transmitted through direct contact with infected body fluids. This type is typically chronic.
    - **Hepatitis D:** Hepatitis D is a secondary infection that only occurs in people infected with Hepatitis B.
    - **Hepatitis E:** Hepatitis E is typically transmitted through consuming contaminated water. It is usually acute and resolves on its own.

    ### Prevention
    - Vaccinations are available for Hepatitis A and B.
    - Practice good hygiene.
    - Avoid sharing needles or other personal items.
    - Use condoms during sexual intercourse.
    - Avoid consumption of contaminated food or water.
    """
    )


elif page == "Contact us":
    ("""
     ### Get in Touch
    We would love to hear from you! You can reach us through the following channels:

    - **Website:** [www.hepatitis-awareness.org](https://www.hepatitis-awareness.org)
    - **Twitter:** [@HepatitisAware](https://twitter.com/HepatitisAware)
    - **Email:** contact@hepatitis-awareness.org

    Whether you have a question about our resources, need support, or just want to share your thoughts, feel free to reach out!
     """)

    # Video

    video_path="record.mp4"

    video= open(video_path, "rb")
    video_bytes =video.read()

    st.video(video_bytes)





