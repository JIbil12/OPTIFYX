import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from PIL import Image

# Load the trained model and scaler
model = pk.load(open('model.pk1', 'rb'))
scaler = pk.load(open('scaler.pk1', 'rb'))

# Set page layout
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

# Add a title and subtitle
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("**Analyze the sentiment of your movie reviews with this simple app!**")

# Add a text input for the user to input their review
review = st.text_input("Enter your movie review", "")

# Center the button using HTML and CSS
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 18px;
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Predict button and prediction result
if st.button('Predict'):
    if review:
        # Preprocess and predict
        review_scale = scaler.transform([review]).toarray()
        res = model.predict(review_scale)
        
        # Display result with styled messages
        if res[0] == 0:
            st.error("🚫 Negative Review")
        else:
            st.success("✅ Positive Review")
    else:
        st.warning("Please enter a review before clicking Predict!")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        Made with ❤️ by Jibil Joseph
    </div>
    """, unsafe_allow_html=True)
