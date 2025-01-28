import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax




with st.sidebar:
    st.header("About")
    st.write(
        """
        This app uses the **Hugging Face Transformers library** and the **RoBERTa model** for sentiment analysis. 
        Enter text, and the app will classify it as positive, neutral, or negative sentiment.
        """
    )
    st.markdown("[GitHub Repository](https://github.com/Anaspro72/Sentiment-Analyzer-using-RoBERTa-Model)", unsafe_allow_html=True)
    st.write("Explore the source code and contribute to the project!")

# Load the model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load_model()

# Define the polarity scoring function
def polarity_scores_roberta(example):
    encoded_text = tokenizer(
        example,
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=512,
        add_special_tokens=True,
    )
    output = model(**encoded_text)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    scores_dict = {
        "Roberta neg": scores[0],
        "Roberta neu": scores[1],
        "Roberta pos": scores[2],
    }
    return scores_dict

# Streamlit app
st.title("Sentiment Analyzer")
st.write("Analyze the sentiment of your feedback")

# User input
user_input = st.text_area("Enter your feedback:", placeholder="Type your feedback here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        scores = polarity_scores_roberta(user_input)

        # Display scores
        st.subheader("Sentiment Scores:")
        for key, value in scores.items():
            st.write(f"{key}: {value:.4f}")

        # Feedback based on sentiment
        if scores["Roberta pos"] > 0.5:
            st.success("The feedback is positive.")
        elif scores["Roberta neg"] > 0.5:
            st.error("The feedback is negative.")
        else:
            st.info("The feedback is neutral.")
    else:
        st.warning("Please enter some feedback before analyzing.")


# Footer
st.markdown("---")
st.markdown(
    """
    **Sentiment Analyzer**  
    Built with ❤️ using the [Hugging Face Transformers](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) library.  
    Check out the [https://github.com/Anaspro72/Sentiment-Analyzer-using-RoBERTa-Model) for the source code.  
    Created by [Anas Khan](https://www.linkedin.com/in/anas-khan-a05730148/).  
    """
)



