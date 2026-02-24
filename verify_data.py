import pandas as pd

try:
    df = pd.read_csv('scraped_reviews.csv')
    print(f"Total Reviews: {len(df)}")
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    
    if len(df) > 10000:
        print("\nSUCCESS: Dataset has > 10,000 samples.")
    else:
        print(f"\nWARNING: Dataset has only {len(df)} samples (Target > 10,000).")
        
    unique_labels = df['label'].nunique()
    if unique_labels >= 3:
        print(f"SUCCESS: Dataset has {unique_labels} classes.")
    else:
        print(f"WARNING: Dataset has only {unique_labels} classes (Target >= 3).")
        
except Exception as e:
    print(f"Error checking data: {e}")
