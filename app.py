import streamlit as st
import pickle

# Load the saved SVC model and TfidfVectorizer
with open('svc_model.pkl', 'rb') as file:
    svc_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("This app classifies IMDB movie reviews as positive or negative.")

# Input text box
review_text = st.text_area("Enter a movie review")

# Button for prediction
if st.button("Predict Sentiment"):
    # Transform the input text using the loaded TfidfVectorizer
    transformed_text = tfidf_vectorizer.transform([review_text])

    # Make prediction
    prediction = svc_model.predict(transformed_text)

    # Display result
    if prediction == 'positive':
        st.success("The review is Positive! 🎉")
    else:
        st.error("The review is Negative! 😞")
