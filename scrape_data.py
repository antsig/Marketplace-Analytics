from google_play_scraper import Sort, reviews_all, reviews
import pandas as pd
import numpy as np
import time

def scrape_reviews(app_id, lang='id', country='id', count=5000):
    print(f"Scraping {count} reviews for {app_id}...")
    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=count
        )
        return result
    except Exception as e:
        print(f"Error scraping {app_id}: {e}")
        return []

# Target Apps
apps = ['com.shopee.id', 'com.tokopedia.tkpd', 'com.instagram.android', 'com.gojek.app']

all_reviews = []
output_file = 'scraped_reviews.csv'

for app in apps:
    print(f"Processing {app}...")
    app_reviews = scrape_reviews(app, count=4000) # Reduced count per app to be safer, added one more app
    if app_reviews:
        for r in app_reviews:
            r['app_id'] = app
        all_reviews.extend(app_reviews)
        print(f"Got {len(app_reviews)} reviews from {app}.")
        
        # Save incrementally
        temp_df = pd.DataFrame(all_reviews)
        # Simple formatting
        if 'content' in temp_df.columns and 'score' in temp_df.columns:
             temp_save = temp_df[['content', 'score', 'app_id']]
             temp_save.to_csv(output_file, index=False)
             print(f"Saved {len(temp_df)} reviews so far to {output_file}")
    
    time.sleep(2) # Be nice to the server

df = pd.DataFrame(all_reviews)

if not df.empty:
    print(f"Total reviews scraped: {len(df)}")
    
    # Simple formatting
    df = df[['content', 'score', 'app_id']]
    
    # Labeling
    def label_score(score):
        if score <= 2:
            return 'Negative'
        elif score == 3:
            return 'Neutral'
        else:
            return 'Positive'

    df['label'] = df['score'].apply(label_score)
    
    # Final Save
    df.to_csv(output_file, index=False)
    print(f"Final data saved to {output_file}")
    
    # Check distribution
    print(df['label'].value_counts())
else:
    print("No reviews found.")
