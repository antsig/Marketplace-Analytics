# Sentiment Analysis with Multi-Layer Perceptron (MLP)

This project demonstrates a complete machine learning pipeline for sentiment analysis using data scraped from the Google Play Store. The pipeline includes data scraping, exploratory data analysis, text preprocessing (TF-IDF), training various Multi-Layer Perceptron (MLP) classification models, optimizing hyperparameters, and saving the best model for inference.

## Project Structure

- `scrape_data.py` - Script to scrape reviews from the Google Play Store and save them to a CSV file.
- `scraped_reviews.csv` - The dataset containing the scraped raw text reviews and their ratings.
- `deep_learning_submission.ipynb` - Jupyter Notebook encompassing the core workflow:
  - Data loading and cleaning
  - Exploratory Data Analysis (EDA)
  - Text Vectorization (TF-IDF)
  - Handling imbalanced data (SMOTE)
  - Training baseline, balanced, and optimized MLP models using `scikit-learn`
  - Evaluating models and plotting results
  - Saving the robust model
- `inference.py` - A command-line script to test the trained classification model on new text inputs.
- `best_sentiment_model_final.pkl` - The serialized form of the best performing NLP pipeline (TF-IDF + MLPClassifier).
- `requirements.txt` - Python dependencies needed to run the files.

## Prerequisites

Make sure you have Python installed. You can install all necessary dependencies using:

```bash
pip install -r requirements.txt
```

### Key Dependencies:

- `google-play-scraper`
- `pandas`, `numpy`
- `scikit-learn`, `imbalanced-learn`
- `matplotlib`, `seaborn`
- `joblib`

## Usage Instructions

### 1. Data Scraping

If you want to gather new data from the Google Play Store, run the scraper script:

```bash
python scrape_data.py
```

This will produce or update the `scraped_reviews.csv` file.

### 2. Model Training

Open the Jupyter Notebook:

```bash
jupyter notebook deep_learning_submission.ipynb
```

Run all cells in the notebook. It will read the CSV dataset, train several permutations of the MLP model, evaluate their performance, and save the best model to `best_sentiment_model_final.pkl`.

### 3. Inference

You can test the trained model against various text inputs directly from your terminal using `inference.py`:

**Testing predefined examples:**

```bash
python inference.py
```

**Testing a custom input:**

```bash
python inference.py "Aplikasi ini sangat bagus dan bermanfaat"
```

The script will load the saved `.pkl` model and output the predicted sentiment class (Negative, Neutral, or Positive).
