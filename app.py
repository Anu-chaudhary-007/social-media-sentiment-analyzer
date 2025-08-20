import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("📊 Social Media Sentiment Analyzer")

# Input text
user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze"):
    if user_input:
        analysis = TextBlob(user_input)
        polarity = analysis.sentiment.polarity

        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Polarity Score:** {polarity:.2f}")

        # Pie Chart with fixed labels & colors
        labels = ["Positive", "Negative", "Neutral"]
        sizes = [
            1 if sentiment == "Positive" else 0,
            1 if sentiment == "Negative" else 0,
            1 if sentiment == "Neutral" else 0,
        ]
        colors = ['green', 'red', 'orange']

        fig1, ax1 = plt.subplots()
        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=[l if s > 0 else "" for l, s in zip(labels, sizes)],  # hide 0% labels
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
            colors=colors
        )
        ax1.axis("equal")
        st.pyplot(fig1)

        # Sentiment Meter (Gauge)
        value = 0
        if sentiment == "Positive":
            value = 80
        elif sentiment == "Neutral":
            value = 50
        elif sentiment == "Negative":
            value = 20

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': f"Sentiment Meter: {sentiment}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 33], 'color': "red"},
                    {'range': [34, 66], 'color': "orange"},
                    {'range': [67, 100], 'color': "green"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))

        st.plotly_chart(gauge)

        # Add Emoji display
        if sentiment == "Positive":
            st.markdown("😊 **Great! People like this.**")
        elif sentiment == "Neutral":
            st.markdown("😐 **It’s okay, neutral vibes.**")
        else:
            st.markdown("😡 **Oops! Negative reaction detected.**")
