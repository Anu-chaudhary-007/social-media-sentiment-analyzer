import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import tweepy

# ---------------- Fail-fast guard for Twitter API ---------------- #
TWITTER_BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", os.getenv("TWITTER_BEARER_TOKEN"))

if not TWITTER_BEARER_TOKEN:
    st.error("❌ Missing TWITTER_BEARER_TOKEN. Add it in Streamlit Secrets or as an env var.")
    st.stop()

# ---------------- Twitter fetch function (Bearer token) ---------------- #
def fetch_tweets(query, count=10):
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

    response = client.search_recent_tweets(
        query=query,
        max_results=min(count, 100),
        tweet_fields=["text", "lang", "created_at"]
    )

    tweets = []
    if response.data:
        for tweet in response.data:
            if tweet.lang == "en":  # only English tweets
                tweets.append(tweet.text)

    return tweets


# ---------------- Streamlit App ---------------- #
st.title("📊 Social Media Sentiment Analyzer")

option = st.radio("Choose input method:", ["Manual Text", "Fetch Tweets"])

# -------- Manual Text Analysis -------- #
if option == "Manual Text":
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

            # Pie Chart
            labels = ["Positive", "Negative", "Neutral"]
            sizes = [
                1 if sentiment == "Positive" else 0,
                1 if sentiment == "Negative" else 0,
                1 if sentiment == "Neutral" else 0,
            ]
            colors = ['green', 'red', 'orange']

            fig1, ax1 = plt.subplots()
            ax1.pie(
                sizes,
                labels=[l if s > 0 else "" for l, s in zip(labels, sizes)],  # hide 0% labels
                autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
                startangle=90,
                colors=colors
            )
            ax1.axis("equal")
            st.pyplot(fig1)

            # Sentiment Meter (Gauge)
            value = {"Positive": 80, "Neutral": 50, "Negative": 20}[sentiment]

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
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

            # Emoji feedback
            if sentiment == "Positive":
                st.markdown("😊 **Great! People like this.**")
            elif sentiment == "Neutral":
                st.markdown("😐 **It’s okay, neutral vibes.**")
            else:
                st.markdown("😡 **Oops! Negative reaction detected.**")


# -------- Fetch Tweets & Analyze -------- #
elif option == "Fetch Tweets":
    query = st.text_input("Enter a keyword or hashtag (e.g., #AI)")
    count = st.slider("Number of tweets to fetch", 5, 50, 10)

    if st.button("Fetch & Analyze Tweets"):
        if query:
            tweets = fetch_tweets(query, count=count)

            if not tweets:
                st.error("No tweets found or API issue.")
            else:
                st.write(f"📌 Showing {len(tweets)} recent tweets:")
                for t in tweets:
                    st.write("-", t)

                # Sentiment analysis for all tweets
                sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
                for t in tweets:
                    analysis = TextBlob(t)
                    polarity = analysis.sentiment.polarity
                    if polarity > 0:
                        sentiments["Positive"] += 1
                    elif polarity < 0:
                        sentiments["Negative"] += 1
                    else:
                        sentiments["Neutral"] += 1

                st.write("### 🧾 Sentiment Summary")
                st.json(sentiments)

                # Pie chart
                fig2, ax2 = plt.subplots()
                ax2.pie(
                    sentiments.values(),
                    labels=sentiments.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['green', 'red', 'orange']
                )
                ax2.axis("equal")
                st.pyplot(fig2)
