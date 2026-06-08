# 🐦 Twitter Sentiment Analytics Dashboard

An interactive NLP-powered Streamlit dashboard that classifies tweet sentiment in real-time using a Logistic Regression model trained with TF-IDF vectorization. Supports single tweet analysis and bulk CSV batch processing with visual analytics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://twitter-sentiment-analytics-dashboard-sybzhk4p9qefdprjqpttpq.streamlit.app/)

---

## 🚀 Live Demo

🔗 **[Try the app here](https://twitter-sentiment-analytics-dashboard-sybzhk4p9qefdprjqpttpq.streamlit.app/)**

Paste any tweet → Get instant Positive / Negative prediction with confidence score

---

## 🎬 Demo

![Twitter Sentiment Dashboard Demo](twitter_sentiment_dashboard_demo.gif)

---

## 📸 Screenshots

<table>
  <tr>
    <td><strong>Dashboard Home</strong><br><img src="assets/dashboard_home.png" width="400"/></td>
    <td><strong>Single Tweet Prediction</strong><br><img src="assets/prediction.png" width="400"/></td>
  </tr>
  <tr>
    <td><strong>Batch Analysis</strong><br><img src="assets/batch_analysis.png" width="400"/></td>
    <td><strong>Batch Prediction Results</strong><br><img src="assets/batch_prediction.png" width="400"/></td>
  </tr>
  <tr>
    <td><strong>Sentiment Distribution</strong><br><img src="assets/pie_chart.png" width="400"/></td>
    <td><strong>Word Cloud</strong><br><img src="assets/word_cloud.png" width="400"/></td>
  </tr>
</table>

---

## ✨ Features

- **Single Tweet Analysis** — paste any tweet, get instant Positive/Negative prediction with confidence score
- **Batch CSV Analysis** — upload a CSV of tweets, get predictions for all rows + downloadable results
- **Sentiment Distribution** — interactive Plotly pie chart of positive vs negative split
- **Word Cloud** — visual representation of most frequent words in the dataset
- **Confidence Score** — model prediction probability shown for every single tweet
- **Downloadable Results** — export batch predictions as a CSV file

---

## 🧠 How It Works

```
Tweet Input
    ↓
Text Preprocessing
(lowercase → remove URLs → remove @mentions → strip punctuation)
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression Model
    ↓
Prediction: Positive 😊 / Negative 😞 + Confidence Score
```

---

## 📊 Model Performance

| Metric     | Score               |
|------------|---------------------|
| Model      | Logistic Regression |
| Vectorizer | TF-IDF              |
| Accuracy   | ~77%                |
| F1 Score   | ~0.76               |
| Dataset    | ~2,000 tweets       |

---

## 🛠️ Tech Stack

| Layer         | Technology                         |
|---------------|------------------------------------|
| Frontend      | Streamlit                          |
| ML Model      | Scikit-learn (Logistic Regression) |
| Vectorizer    | TF-IDF (Scikit-learn)              |
| Visualization | Plotly, WordCloud, Matplotlib      |
| Data          | Pandas, NumPy                      |
| Deployment    | Streamlit Cloud                    |
| Language      | Python 3.10+                       |

---

## 📁 Project Structure

```
Twitter-Sentiment-Analytics-Dashboard/
├── app.py                                  # Main Streamlit application
├── sentiment_model.pkl                     # Trained Logistic Regression model
├── tfidf_vectorizer.pkl                    # Fitted TF-IDF vectorizer
├── sample_tweets.csv                       # Sample CSV for testing batch analysis
├── requirements.txt                        # Python dependencies
├── twitter_sentiment_dashboard_demo.gif    # Demo GIF
├── assets/                                 # Dashboard screenshots
│   ├── dashboard_home.png
│   ├── prediction.png
│   ├── batch_analysis.png
│   ├── batch_prediction.png
│   ├── pie_chart.png
│   └── word_cloud.png
└── notebooks/
    └── Twitter_Sentiment_Analysis.ipynb    # Model training notebook
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/varun0852/Twitter-Sentiment-Analytics-Dashboard.git
cd Twitter-Sentiment-Analytics-Dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 🧪 Try It Out

For **single tweet** analysis, paste any of these:
- `I love this product!` → Positive 😊
- `Worst experience ever` → Negative 😞
- `Amazing customer support` → Positive 😊
- `This app keeps crashing` → Negative 😞

For **batch analysis**, upload any CSV with a column named `tweet`.
A `sample_tweets.csv` is included in the repo to test with.

---

## 👤 Author

**Varun** — AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/varun-a87781274/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/varun0852)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:diwakarvarun752@gmail.com)
