import streamlit as st
import pandas as pd
from src.sentiment import analyze_sentiment

st.title("🔍 Social Media Sentiment Analyzer")

option = st.radio("Choose input type:", ["Text", "CSV File"])

if option == "Text":
    user_input = st.text_area("Enter text:")
    if st.button("Analyze"):
        if user_input.strip():
            result = analyze_sentiment(user_input)
            st.write("### 📊 Analysis Result")
            st.write(f"**Text:** {result['text']}")
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Score:** {result['score']:.2%}")
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must have a column named 'text'")
        else:
            df["Sentiment"] = df["text"].apply(
                lambda x: analyze_sentiment(x)["label"]
            )
            st.write("### 📊 Results")
            st.dataframe(df)

