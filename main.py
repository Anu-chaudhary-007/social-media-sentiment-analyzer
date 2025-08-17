import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Use CPU if GPU not available
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)


def analyze_sentiment(text):
    """Analyze sentiment for a single text"""
    result = pipeline(text)[0]
    return {
        "text": text,
        "label": result["label"],
        "score": round(result["score"] * 100, 2)
    }


def analyze_csv(file_path):
    """Analyze sentiment for a CSV file and show chart"""
    try:
        df = pd.read_csv(file_path)

        if 'text' not in df.columns:
            print("❌ CSV must have a column named 'text'")
            return

        results = []
        for txt in df['text']:
            sentiment = analyze_sentiment(str(txt))
            results.append(sentiment)

        result_df = pd.DataFrame(results)
        output_path = "output_sentiment.csv"
        result_df.to_csv(output_path, index=False)

        print(f"✅ Analysis complete! Results saved to {output_path}")

        # --- Visualization ---
        counts = result_df['label'].value_counts()
        counts.plot(kind='bar', color=['green', 'red', 'blue'])
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.show()

    except Exception as e:
        print(f"⚠️ Error processing CSV: {e}")


if __name__ == "__main__":
    print("🔍 Social Media Sentiment Analyzer")
    user_input = input("Enter text or CSV file path: ")

    if os.path.isfile(user_input) and user_input.endswith(".csv"):
        analyze_csv(user_input)
    else:
        result = analyze_sentiment(user_input)
        print("\n📊 Analysis Result")
        print("------------------------------")
        print(f"Text   : {result['text']}")
        print(f"Label  : {result['label']} {'✅' if result['label']=='POSITIVE' else '❌'}")
        print(f"Score  : {result['score']}%")


