import joblib
import os

# Use relative paths so it works anywhere
MODEL_PATH = os.path.join('models', 'logistic_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

def load_model():
    """Loads and returns the trained Logistic Regression model."""
    model = joblib.load(MODEL_PATH)
    return model

def load_vectorizer():
    """Loads and returns the fitted TF-IDF vectorizer."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    return vectorizer
