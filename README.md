# 📰 Fake News Detection Web App

This project is a Fake News Detection system developed using **Machine Learning** and **Deep Learning** techniques. It is built as a web application using **Flask**, allowing users to classify news articles as *real* or *fake*.

---

## Features

- 🔍 Detects fake vs real news using two models:
-  **XGBoost with TF-IDF**
-  **BERT (Bidirectional Encoder Representations from Transformers)**
-  Shows model performance with confusion matrices
-  User login authentication (default: `adhil` / `1234`)
-  Modern UI/UX with dark mode and animated design
- Logging system to track predictions

---

## Models Used

### 1. **XGBoost + TF-IDF**
- Fast and lightweight
- Preprocessed using stopword removal and lemmatization
- Trained with GridSearchCV hyperparameter tuning

### 2. **BERT (Fine-tuned)**
- Deep learning model for better semantic understanding
- Fine-tuned on reduced fake/true datasets
- Offers better generalization on unseen news

---

## Tech Stack

- **Frontend:** HTML, CSS, JS
- **Backend:** Python (Flask)
- **ML Libraries:** scikit-learn, XGBoost, Transformers
- **Data:** Cleaned and reduced CSV datasets

---

## 📂 Project Structure

```
FAKE NEWS/
│
├── app/
│   ├── __init__.py
│   ├── project.py          # Core ML logic
│   └── routes.py           # Routes for web app
│
├── data/
│   ├── Cleaned_Fake.csv
│   └── Cleaned_True.csv
│
├── models/
│   ├── bert_confusion_matrix.png
│   └── xgboost_confusion_matrix.png
│
├── static/
│   ├── style.css
│   └── image.jpg           # Background image
│
├── templates/
│   ├── index.html          # Main prediction page
│   └── login.html          # Login interface
│
├── fake-news-env/          # Virtual environment (ignored by Git)
│
├── app.py                  # App entry point
├── run.py                  # Runs the server
├── app.log                 # Log file
├── requirements.txt        # Required packages
└── README.md               # Project overview (this file)
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

# Activate your virtual environment
source fake-news-env/bin/activate     # Linux/Mac
# OR
fake-news-env\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python run.py
```

---

## 🔐 Login Details

```bash
Username: adhil
Password: 1234
```

---

## Sample UI

![screenshot](static/image.jpg)

---

## Author

**Adhil mohammed KT**,  
B.Tech Artificial Intelligence & Data Science,  
MES College of Engineering

---

## License

This project is for educational and non-commercial use only.
