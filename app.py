import streamlit as st
import pickle

# ✅ DEBUG START
st.write("App started")

st.write("Loading model...")
model = pickle.load(open("model.pkl", "rb"))

st.write("Loading vectorizer...")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.write("Loaded successfully ✅")
# ✅ DEBUG END
import re

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

# Prediction function
def predict_review(text):
    text_clean = clean_text(text)
    text_vec = vectorizer.transform([text_clean])

    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]
    confidence = prob[prediction] * 100

    if prediction == 1:
        return f"Fake Review ❌ ({confidence:.2f}%)"
    else:
        return f"Real Review ✅ ({confidence:.2f}%)"


# UI
st.title("Fake Review Detection System")

review = st.text_area("Enter your review")

if st.button("Check Review"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        result = predict_review(review)

        if "Fake" in result:
            st.error(result)
        else:
            st.success(result)
