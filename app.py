import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import tweepy
import requests

# ---------------- Fail-fast guard for Twitter API ---------------- #
TWITTER_BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", os.getenv("TWITTER_BEARER_TOKEN"))

if not TWITTER_BEARER_TOKEN:
    st.error("❌ Missing TWITTER_BEARER_TOKEN. Add it in Streamlit Secrets or as an env var.")
    st.stop()

# ---------------- Hugging Face API ---------------- #
HF_API_URL = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis"
HF_HEADERS = {"Authorization": "Bearer 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"}   # your token

def analyze_sentiment(text):
    """Use Hugging Face API for sentiment analysis"""
    try:
        payload = {"inputs": text}
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

        if response.status_code != 200:
            return "Error", 0.0

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            best = max(result[0], key=lambda x: x['score'])
            label = best["label"]
            score = best["score"]

            label_map = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral"}
            return label_map.get(label, label), score
        else:
            return "Error", 0.0
    except Exception as e:
        return "Error", 0.0


# ---------------- Twitter fetch function ---------------- #
def fetch_tweets(query, count=10):
    """Fetch recent tweets using Twitter API v2"""
    try:
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

        return tweets, None
    except Exception as e:
        return [], f"⚠️ Error fetching tweets: {str(e)}"


# ---------------- Visualization Helpers ---------------- #
def plot_sentiment_pie(sentiment):
    labels = ["Positive", "Negative", "Neutral"]
    sizes = [1 if sentiment == l else 0 for l in labels]
    colors = ["green", "red", "orange"]

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=[l if s > 0 else "" for l, s in zip(labels, sizes)],
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        startangle=90,
        colors=colors
    )
    ax.axis("equal")
    return fig


def plot_gauge(sentiment):
    value = {"Positive": 80, "Neutral": 50, "Negative": 20}.get(sentiment, 50)

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
    return gauge


# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="📊 Social Media Sentiment Analyzer", page_icon="📈", layout="centered")
st.title("📊 Social Media Sentiment Analyzer")
st.caption("Analyze sentiments from manual text or live tweets (via Twitter API & Hugging Face).")

option = st.radio("Choose input method:", ["Manual Text", "Fetch Tweets"])

# -------- Manual Text Analysis -------- #
if option == "Manual Text":
    user_input = st.text_area("✍️ Enter text to analyze:")

    if st.button("🔍 Analyze"):
        if user_input.strip():
            sentiment, score = analyze_sentiment(user_input)

            st.subheader("📌 Sentiment Result")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence Score:** {score:.2f}")

            # Visuals
            if sentiment in ["Positive", "Negative", "Neutral"]:
                st.pyplot(plot_sentiment_pie(sentiment))
                st.plotly_chart(plot_gauge(sentiment))

                # Emoji Feedback
                emoji_map = {
                    "Positive": "😊 **Great! People like this.**",
                    "Neutral": "😐 **It’s okay, neutral vibes.**",
                    "Negative": "😡 **Oops! Negative reaction detected.**"
                }
                st.markdown(emoji_map[sentiment])
        else:
            st.warning("⚠️ Please enter some text.")


# -------- Fetch Tweets & Analyze -------- #
elif option == "Fetch Tweets":
    query = st.text_input("🔑 Enter a keyword or hashtag (e.g., #AI)")
    count = st.slider("Number of tweets to fetch", 5, 50, 10)

    if st.button("📥 Fetch & Analyze Tweets"):
        if not query.strip():
            st.warning("⚠️ Please enter a valid keyword or hashtag.")
        else:
            with st.spinner("Fetching tweets... ⏳"):
                tweets, error = fetch_tweets(query, count)

            if error:
                st.error(error)
            elif not tweets:
                st.error("⚠️ No tweets found for this query.")
            else:
                st.success(f"✅ Fetched {len(tweets)} recent tweets for '{query}'")

                st.subheader("📌 Sample Tweets")
                for i, t in enumerate(tweets[:5], 1):
                    st.write(f"**Tweet {i}:** {t}")

                # Sentiment analysis for all tweets
                sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
                for t in tweets:
                    s, _ = analyze_sentiment(t)
                    if s in sentiments:
                        sentiments[s] += 1

                st.subheader("🧾 Sentiment Summary")
                st.json(sentiments)

                # Pie Chart
                fig2, ax2 = plt.subplots()
                ax2.pie(
                    sentiments.values(),
                    labels=sentiments.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=["green", "red", "orange"]
                )
                ax2.axis("equal")
                st.pyplot(fig2)
