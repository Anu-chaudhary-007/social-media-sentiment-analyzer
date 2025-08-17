import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Load sentiment analysis pipeline (CPU only)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1
)

st.title("📊 Social Media Sentiment Analyzer")
st.write("This app analyzes the sentiment of your text (Positive, Negative, Neutral).")

# Text input
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Split input into sentences
        sentences = [s.strip() for s in user_input.replace("?", ".").replace("!", ".").split(".") if s.strip()]

        results = sentiment_pipeline(sentences)

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for sentence, result in zip(sentences, results):
            label = result["label"].lower()
            score = round(result["score"], 4)

            if label == "positive":
                color = "#4CAF50"  # green
            elif label == "negative":
                color = "#F44336"  # red
            else:
                color = "#FF9800"  # orange (neutral)

            # Update counts
            if label in sentiment_counts:
                sentiment_counts[label] += 1

            # Show each sentence result
            st.markdown(f"**Sentence:** {sentence}")
            st.markdown(f"<h4 style='color:{color};'>Sentiment: {label.capitalize()}</h4>", unsafe_allow_html=True)

            # Custom progress bar
            progress_html = f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; width: 100%; height: 20px;">
                <div style="background-color:{color}; width:{score*100}%; height:100%; border-radius: 8px;"></div>
            </div>
            <p style="text-align:center;">Confidence: {score*100:.1f}%</p>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
            st.write("---")

        # Show pie chart summary
        st.subheader("📈 Overall Sentiment Distribution")
        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts.values(),
            labels=[k.capitalize() for k in sentiment_counts.keys()],
            autopct='%1.1f%%',
            colors=["#4CAF50", "#F44336", "#FF9800"]
        )
        ax.axis("equal")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please enter some text to analyze.")
