Sentiment Analyzer

A simple Sentiment Analysis Web App built with Streamlit and Hugging Face Transformers.
It classifies text into Positive, Negative, or Neutral sentiments.







🚀 Features

Analyze the sentiment of any text input.

Supports Positive, Negative, and Neutral classifications.

Built with Streamlit for an interactive UI.

Uses Hugging Face model: finiteautomata/bertweet-base-sentiment-analysis
.

Can run with Hugging Face Inference API (using your API token).






🛠️ Installation

Clone the repository:

git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer





Install dependencies:

pip install -r requirements.txt





🔑 Hugging Face API Setup

Create a free account on Hugging Face
.

Go to Settings → Access Tokens.

Create a Read token.

Set the token as an environment variable:





Linux / macOS:

export HF_API_TOKEN="your_token_here"


Windows (PowerShell):

setx HF_API_TOKEN "your_token_here"




▶️ Usage

Run the Streamlit app:

streamlit run app.py


Then open http://localhost:8501
 in your browser.





📌 Example

Input:

I love this project, it works great!


Output:

Sentiment: Positive ✅




    
📄 Requirements

See requirements.txt







🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.






📜 License

This project is licensed under the MIT License.
