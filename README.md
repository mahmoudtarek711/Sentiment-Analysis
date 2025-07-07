# 📊 Sentiment Analysis API with Logistic Regression & FastAPI

A simple text sentiment analysis pipeline:
- **Preprocessing** with custom cleaning & stopwords removal
- **TF-IDF Vectorizer**
- **Logistic Regression** (tuned with GridSearchCV)
- **FastAPI** server for real-time predictions

---

## 📁 **Project Structure**

```
your_project/
│
├── app/
│   ├── __init__.py         # Makes app/ a Python package
│   ├── api.py              # FastAPI server with /predict endpoint
│   ├── model.py            # Loads trained model & vectorizer
│   ├── preprocess.py       # Cleaning + vectorizing pipeline
│
├── models/
│   ├── logistic_model.pkl  # Saved Logistic Regression model
│   ├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
│
├── data/                   # Your raw/cleaned datasets (optional)
│
├── train.py                # Training script with GridSearchCV
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

---

## ✅ **Installation**

1. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ✅ **Training the Model**

1. Make sure your `train.py`:
   - Loads your data
   - Cleans the text using `app/preprocess.py`
   - Vectorizes with `TfidfVectorizer`
   - Tunes hyperparameters with `GridSearchCV`
   - Saves:
     - `logistic_model.pkl`
     - `tfidf_vectorizer.pkl`  
   into the `models/` folder.

2. Run:
   ```bash
   python train.py
   ```

---

## ✅ **Running the API**

1. Make sure `models/logistic_model.pkl` and `models/tfidf_vectorizer.pkl` exist.

2. Run the FastAPI server:
   ```bash
   uvicorn app.api:app --reload
   ```

3. Open your browser:
   - 📌 **Root:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - 📄 **Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Interactive UI!)

---

## ✅ **API Usage**

### ➜ `POST /predict`

**Request Body**
```json
{
  "text": "I absolutely loved this product!"
}
```

**Response**
```json
{
  "input_text": "I absolutely loved this product!",
  "predicted_class": 1,
  "sentiment_label": "Positive"
}
```

---

## ✅ **Notes**

- **Preprocessing:** The cleaning logic (`clean_text()`) lives in `app/preprocess.py`. It removes mentions, punctuation, lowercases text, and removes stopwords.  
  Your `train.py` **must use the same cleaning** for consistency!

- **Model Loading:** `app/model.py` loads the Logistic Regression model & TF-IDF vectorizer saved by `train.py`.

- **Inference:** `app/api.py` wraps everything in a FastAPI server. Use Swagger docs to test it!

- **No `main.py` needed:** In production, just run via FastAPI & Swagger.

---

## ✅ **Requirements**

Example `requirements.txt`:
```
fastapi>=0.95
uvicorn[standard]>=0.22
scikit-learn>=1.0
pandas>=1.4
numpy>=1.21
joblib>=1.2
```

---

## ✅ **Run Checklist**

✅ `models/` contains `logistic_model.pkl` and `tfidf_vectorizer.pkl`  
✅ `app/__init__.py` exists (makes `app/` a package)  
✅ Imports in `api.py` use **relative imports**, e.g. `from .model import load_model`  
✅ Run `uvicorn` from the **project root**, not inside `app/`
