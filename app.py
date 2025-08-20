import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob

# Streamlit app title
st.title("Social Media Sentiment Analyzer")

# Text input
user_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Sentiment analysis using TextBlob
        analysis = TextBlob(user_input)
        polarity = analysis.sentiment.polarity

        # Determine sentiment category
        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Show results
        st.subheader("🔍 Sentiment Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity Score:** {polarity:.2f}")

        # Sentiment counts (for pie chart demo, single input shown as one category)
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        sentiment_counts[sentiment] += 1

        # Pie chart for sentiment distribution
        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts.values(),
            labels=sentiment_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,           # rotate chart for better alignment
            pctdistance=0.85,        # move percentages closer to center
            labeldistance=1.1        # move labels slightly outward
        )
        ax.axis('equal')  # Equal aspect ratio ensures a perfect circle
        st.pyplot(fig)
