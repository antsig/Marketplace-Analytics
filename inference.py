import joblib
import sys

MODEL_FILE = 'best_sentiment_model_final.pkl'

def predict_sentiment(text):
    try:
        model = joblib.load(MODEL_FILE)
        prediction_idx = model.predict([text])[0]
        # Hardcoded mapping based on sorted unique labels
        # 0=Negative, 1=Neutral, 2=Positive
        labels = ['Negative', 'Neutral', 'Positive']
        return labels[prediction_idx]
    except FileNotFoundError:
        return "Model file not found. Please train the model first."
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("--- Sentiment Analysis Inference ---")
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Input: {text}")
        print(f"Sentiment: {predict_sentiment(text)}")
    else:
        # Demo inputs
        examples = [
            "Aplikasi ini sangat membantu, fitur lengkap!",
            "Sering crash saat dibuka, tolong diperbaiki.",
            "Biasa aja sih, standar aplikasi belanja.",
            "Barang sangat bagus, pengiriman cepat mantap!",
            "Kualitas produk hancur, saya sangat kecewa. Jangan beli disini.",
            "Lumayan lah untuk harga segitu, standar aja.",
            "Barang tidak sesuai pesanan, parah banget",
            "Suka banget sama produknya, bakal langganan"
        ]
        
        try:
            model = joblib.load(MODEL_FILE)
            print("Model loaded successfully.")
            
            for ex in examples:
                pred = model.predict([ex])[0]
                # Hardcoded mapping based on sorted unique labels
                print(f"Text: '{ex}'\nSentiment: {pred}\n")# 0=Negative, 1=Neutral, 2=Positive
                
        except FileNotFoundError:
            print(f"Error: {MODEL_FILE} not found. Run deep_learning_submission.ipynb first.")
