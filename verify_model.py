import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Config ---
DATA_FILE = 'scraped_reviews.csv'
MODEL_FILE = 'best_sentiment_model_final.pkl'

# --- 1. Load & Clean Data ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep letters only
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(.)\1{2,}', r'\1', text) # Normalize repeating chars
    return text

def verify():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df['content'] = df['content'].fillna('').astype(str)
    df['content_clean'] = df['content'].apply(clean_text)
    
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    X = df['content_clean']
    y = df['label_encoded']
    
    # Split (Same seed as training to ensure same test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print("Loading model...")
    model = joblib.load(MODEL_FILE)
    
    print("Predicting...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    verify()
