import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load saved model
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Page Config

st.set_page_config(
    page_title="Twitter Sentiment Analytics Dashboard",
    page_icon="🐦"
)
# Sidebar

st.sidebar.title("About")

st.sidebar.info("""
Twitter Sentiment Analytics Dashboard

Built using:
- NLP
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

Author: Varun
""")

# Title

st.title("🐦 Twitter Sentiment Analytics Dashboard")

# Metrics

col1, col2 = st.columns(2)

with col1:
    st.metric("Model" , "Logistic Regression")

with col2:
        st.metric("Vectorizer" , "TF-IDF")

# Tabs

tab1, tab2 = st.tabs([
     "Single Tweet Analysis",
     "Batch Analysis"
])

# Tab1 SINGLE TWEET ANALYSIS

with tab1:
     
     st.subheader("Analyze Tweet Sentiment")

     st.info("""
Try Examples:

• I love this product
• Worst experience ever
• Amazing customer support
• This app is terrible
""")

     tweet = st.text_area(
          "Enter a Tweet",
          height = 140
     )
     if st.button("Analyze Sentiment"):
          cleaned = clean_text(tweet)
          vector = tfidf.transform([cleaned])
          prediction = model.predict(vector)[0]
          confidence = model.predict_proba(vector).max()

          st.subheader("Prediction")

          if prediction == 1:
               st.success("😊 Positive Sentiment")
               st.balloons()
          else:
               st.error("😞 Negative Sentiment")
          st.metric(
                 "Confidence Score",
                 f"{confidence:.2%}"
            )

# Tab2 BATCH ANALYSIS

with tab2:
     
     st.subheader("Batch Sentiment Analysis")
     uploaded_file = st.file_uploader(
          "Upload CSV File",
          type=["csv"]
     )

     if uploaded_file is not None:
          df = pd.read_csv(uploaded_file)

          st.write("Dataset Preview")

          st.dataframe(df.head())

          # Important CSV must contain a column named "tweet"

          # Clean Text 

          df["cleaned"] = df["tweet"].apply(clean_text)
          
          # Transform 

          vectors = tfidf.transform(df["cleaned"])

          # Predict

          predictions = model.predict(vectors)

          df["sentiment"] = predictions

          df["sentiment"] = df["sentiment"].map({
               1: "Positive",
               0: "Negative"
          })

          st.write("Prediction Results")

          display_df = df[["tweet", "sentiment"]]
          st.dataframe(display_df)

          # Statistics

          positive_count = (
               df["sentiment"] == "Positive"
          ).sum()

          negative_count = (
               df["sentiment"] == "Negative"
          ).sum()

          col1, col2 = st.columns(2)

          with col1:
               st.metric(
                    "Positive Tweets",
                    positive_count
               )
          with col2:
                st.metric(
                     "Negative Tweets",
                     negative_count
                )

          # Pie Chart

          st.write("Sentiment Distribution")

          fig = px.pie(
                df,
                names="sentiment",
                title="Sentiment Distribution",
                color="sentiment",
                color_discrete_map={
                    "Positive": "#00CC96",
                    "Negative": "#EF553B"
                }
            )

          st.plotly_chart(fig, use_container_width=True)

          # Word Cloud
          st.write(" Word Cloud")

          text = " ".join(df["tweet"].astype(str))

          wordcloud = WordCloud(
               width = 800,
               height = 400,
               background_color = "white"
          ).generate(text)

          fig_wc, ax = plt.subplots()

          ax.imshow(wordcloud)
          ax.axis("off")

          st.pyplot(fig_wc)

          # Download Results
          csv = df.to_csv(index=False)

          st.download_button(
               label="Download Results",
               data=csv,
               file_name="sentiment_results.csv",
               mime="text/csv"
          )

# Footer
st.markdown("---")

st.markdown(
    """
    <center>
    Built by <b>Varun</b> • NLP • Machine Learning • Streamlit
    </center>
    """,
    unsafe_allow_html=True
) 