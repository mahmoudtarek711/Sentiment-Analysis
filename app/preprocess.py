import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Use scikit-learn's built-in stop words or your custom list
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text: str) -> str:
    """
    Apply the same cleaning used during training.
    1. Lowercase
    2. Remove mentions (@username)
    3. Remove punctuation
    4. Remove stopwords
    """
    text = text.lower()
    text = re.sub(r'@\w+', '', text)              # Remove mentions like @user
    text = re.sub(r'[^\w\s]', '', text)           # Remove punctuation
    tokens = text.split()                         # Tokenize by whitespace
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess(text_list, vectorizer):
    """
    Full preprocessing pipeline for new texts:
    - Apply clean_text to each text
    - Vectorize using the fitted vectorizer
    """
    cleaned_texts = [clean_text(text) for text in text_list]
    return vectorizer.transform(cleaned_texts)
