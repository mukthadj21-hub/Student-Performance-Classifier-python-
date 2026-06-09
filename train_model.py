import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Loading datasets...")
    try:
        df = pd.read_csv('student_data.csv')
    except FileNotFoundError:
        print("Data file not found. Please run generate_data.py first.")
        return
    
    # Feature matrix X and target vector y
    X = df[['Study_Hours', 'Attendance', 'Marks', 'Assignment_Score']]
    y = df['Result']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Pipeline (Scaler + Classifier)
    print("\n--- Training Premium Random Forest Pipeline ---")
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Fitting model...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, rf_preds)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Saving the Pipeline
    print("\nSaving Random Forest Pipeline to rf_model.pkl...")
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_pipeline, f)
        
    print("Model saved successfully!")

if __name__ == '__main__':
    train_and_evaluate()
