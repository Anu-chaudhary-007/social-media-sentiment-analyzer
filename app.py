import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Load sentiment analysis model (supports pos/neg/neutral)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Streamlit App
st.title("🌐 Web App Sentiment Analyzer")
st.write("Enter text and see if it's Positive, Negative, or Neutral with confidence scores.")

text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    results = sentiment_pipeline(text)

    # Extract labels and scores
    labels = [res['label'] for res in results]
    scores = [res['score'] for res in results]

    # Show results
    st.subheader("📊 Analysis Result")
    for label, score in zip(labels, scores):
        st.write(f"**{label}:** {score*100:.2f}%")

    # --- Visualization ---
    st.subheader("📈 Sentiment Confidence")
    fig, ax = plt.subplots()
    ax.bar(labels, scores, color=["green", "red", "gray"])
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)

    # Add percentages on bars
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")

    st.pyplot(fig)


