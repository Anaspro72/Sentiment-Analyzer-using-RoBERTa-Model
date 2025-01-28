# Sentiment Analyzer using RoBERTa Model

This repository contains a simple and interactive **Sentiment Analysis App** built with **Streamlit**. The app uses the **Hugging Face Transformers library** and a pre-trained **RoBERTa model** to classify the sentiment of input text as positive, neutral, or negative.

## Features

- **User-Friendly Interface**: A clean and interactive UI powered by Streamlit.
- **RoBERTa Model**: Sentiment analysis using the pre-trained `cardiffnlp/twitter-roberta-base-sentiment` model.
- **Real-Time Analysis**: Analyze sentiment scores instantly with a detailed breakdown.
- **Interpretation**: Displays whether the feedback is positive, neutral, or negative.
- **Customizable**: The app is fully customizable for various NLP tasks.

## Installation

Follow these steps to set up the project locally:

### Prerequisites

- Python 3.8 or above
- pip (Python package manager)

### Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

This will start a local server. Open your browser and navigate to `http://localhost:8501` to use the app.

## File Structure

```plaintext
sentiment-analyzer/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

## How It Works

1. **Model Setup**: The app uses the `AutoTokenizer` and `TFAutoModelForSequenceClassification` from Hugging Face to load the `cardiffnlp/twitter-roberta-base-sentiment` model.
2. **Sentiment Analysis**: The text input is tokenized, passed through the model, and sentiment probabilities are calculated using the softmax function.
3. **Interactive UI**: The user enters text, clicks the "Analyze Sentiment" button, and the app displays:
   - Sentiment scores (negative, neutral, positive).
   - A conclusion based on the highest score.

## Example Output

Input:

```plaintext
"That's awesome!"
```

Output:

```plaintext
Roberta neg: 0.007432471960783005
Roberta neu: 0.11426050961017609
Roberta pos: 0.878307044506073
The feedback is Positive! 🎉
```

## Dependencies

The app relies on the following libraries:

- `streamlit`
- `transformers`
- `tensorflow`
- `scipy`

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the `cardiffnlp/twitter-roberta-base-sentiment` model.
- Streamlit for providing an amazing framework for building interactive apps.

## Contact

For questions or feedback, feel free to reach out:

- **Email**: [anask8726@gmail.com](mailto:anask8726@gmail.com)
- **GitHub**: [Your GitHub Profile](https://github.com/Anaspro72)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/anas-khan-a05730148/)
