import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="wide")

st.title("🔍 Social Media Sentiment Analyzer")

# Choose input type
input_type = st.radio("Select Input Type:", ["Single Text", "CSV File"])

if input_type == "Single Text":
    text = st.text_area("Enter text here:")
    if st.button("Analyze"):
        if text.strip() != "":
            result = sentiment_analyzer(text)[0]
            st.subheader("📊 Analysis Result")
            st.write(f"**Text**   : {text}")
            st.write(f"**Label**  : {result['label']}")
            st.write(f"**Score**  : {result['score']*100:.2f}%")
        else:
            st.warning("⚠️ Please enter some text.")

elif input_type == "CSV File":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("CSV must have a column named **text** ❌")
        else:
            st.write("📄 Preview of Uploaded File")
            st.dataframe(df.head())

            if st.button("Analyze All"):
                # Run sentiment analysis on all rows
                results = sentiment_analyzer(df["text"].tolist())

                df["label"] = [r["label"] for r in results]
                df["score"] = [r["score"] for r in results]

                st.subheader("📊 Analysis Results")
                st.dataframe(df)

                # Sentiment distribution
                sentiment_counts = df["label"].value_counts()

                fig, ax = plt.subplots()
                sentiment_counts.plot.pie(autopct="%.2f%%", ax=ax, ylabel="")
                st.pyplot(fig)

                # Download option
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )



