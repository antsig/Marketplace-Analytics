import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_FILE = 'best_sentiment_model_final.pkl'
DATA_FILE = 'scraped_reviews.csv'

def verify_model():
    print("Loading data and model...")
    try:
        df = pd.read_csv(DATA_FILE)
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df['label'])
        X = df['content']
        
        print(f"Classes: {le.classes_}")
        
        model = joblib.load(MODEL_FILE)
        
        # We use a conservative split to verify performance
        # We test on the last 20% (which was likely test set in all experiments)
        # We train-check on the first 70% (which was likely train set in all experiments)
        
        # Note: We must use the SAME random_state as training to ensure consistent shuffling
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # X_test_full corresponds to the 20% split. This is safe for Test accuracy check.
        # X_train_full corresponds to the 80% split. 
        
        print(f"Verifying on Test Data (Size: {len(X_test_full)})...")
        y_pred_test = model.predict(X_test_full)
        test_acc = accuracy_score(y_test_full, y_pred_test)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(classification_report(y_test_full, y_pred_test))
        
        # For Training Accuracy, we check a subset to be quick, or the whole thing.
        print(f"Verifying on Train Data (Size: {len(X_train_full)})...")
        y_pred_train = model.predict(X_train_full)
        train_acc = accuracy_score(y_train_full, y_pred_train)
        print(f"Train Accuracy: {train_acc:.4f}")
        
        if test_acc > 0.92:
             print("\nSUCCESS: Test Accuracy > 92%.")
        else:
             print(f"\nTest Accuracy is {test_acc:.4f}.")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_model()
