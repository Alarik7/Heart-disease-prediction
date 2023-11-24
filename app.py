import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to load the dataset
def load_dataset():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

def main():
    # Streamlit app title
    st.title("Heart Disease Prediction and Visualization App")

    # Load the dataset
    heart_data = load_dataset()

    if heart_data is not None:
        st.write("Dataset loaded successfully!")

        # Display basic statistics of the dataset
        st.header("Basic Dataset Statistics:")
        st.write(heart_data.describe())

        # Display all features
        st.header("Dataset Features:")
        st.write(heart_data)

        # Declare the model outside the conditional block
        model = None

        # Sidebar for model training and visualization
        st.sidebar.header("Train Logistic Regression Model")
        if st.sidebar.checkbox("Train Model"):
            X = heart_data.drop(columns='target', axis=1)
            Y = heart_data['target']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

            model = LogisticRegression()
            model.fit(X_train, Y_train)

            st.sidebar.write('Model trained successfully!')

        st.sidebar.header("User Input for Prediction")
        st.sidebar.write("Enter the following information for prediction:")

        age = st.sidebar.slider("Age", 20, 80, 40)
        sex = st.sidebar.radio("Sex", ["Male", "Female"])
        cp = st.sidebar.slider("Chest Pain Type (cp)", 0, 3, 1)
        trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
        chol = st.sidebar.slider("Cholesterol (chol)", 100, 600, 200)
        fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", ["Yes", "No"])
        restecg = st.sidebar.slider("Resting Electrocardiographic Results (restecg)", 0, 2, 0)
        thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", 70, 220, 150)
        exang = st.sidebar.radio("Exercise-Induced Angina (exang)", ["Yes", "No"])
        oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest (oldpeak)", 0.0, 6.2, 1.0)
        slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (slope)", 0, 2, 1)
        ca = st.sidebar.slider("Number of Major Vessels Colored by Flouroscopy (ca)", 0, 3, 0)
        thal = st.sidebar.slider("Thalassemia (thal)", 0, 2, 1)

        # Convert user input to the format expected by the model
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        # Make a prediction only if the model is trained
        if model is not None:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            prediction = model.predict([user_input])

            st.sidebar.write("Prediction Result:")
            if prediction[0] == 0:
                st.sidebar.write("No Heart Disease")
            else:
                st.sidebar.write("Heart Disease")

                # Model evaluation
                X_train_prediction = model.predict(X_train)
                training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
                st.write('Accuracy on Training data : ', training_data_accuracy)

                X_test_prediction = model.predict(X_test)
                test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
                st.write('Accuracy on Test data : ', test_data_accuracy)

                # Confusion matrix and classification report
                confusion = confusion_matrix(Y_train, X_train_prediction)
                st.write("Confusion Matrix (Training)\n", confusion)

                report = classification_report(Y_train, X_train_prediction)
                st.write("Classification Report (Training)\n", report)

        # Sidebar for visualization
        st.sidebar.header("Data Visualization")
        if st.sidebar.checkbox("Show S-Curve"):
            x_values = np.linspace(-10, 10, 1000)
            logistic_values = 1 / (1 + np.exp(-x_values))

            plt.figure(figsize=(8, 6))
            plt.plot(x_values, logistic_values, label='Logistic Function', color='blue')
            plt.title('S-Curve for Logistic Regression')
            plt.xlabel('X-values')
            plt.ylabel('Logistic Values')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

        if st.sidebar.checkbox("Show Pie Chart"):
            class_counts = heart_data['target'].value_counts()
            labels = ['No Heart Disease', 'Heart Disease']
            sizes = class_counts.values
            colors = ['lightcoral', 'lightskyblue']

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.title('Distribution of Heart Disease')
            plt.axis('equal')
            st.pyplot(plt)

        if st.sidebar.checkbox("Show Bar Graph"):
            age_groups = pd.cut(heart_data['age'], bins=[29, 40, 50, 60, 100], labels=['30-39', '40-49', '50-59', '60+'])
            age_group_counts = age_groups.value_counts()

            plt.figure(figsize=(8, 6))
            age_group_counts.plot(kind='bar', color='skyblue')
            plt.title('Distribution of Age Groups')
            plt.xlabel('Age Groups')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(plt)

if __name__ == '__main__':
    main()
