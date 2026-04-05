import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

# Prediction function
def predict_review(text):
    text_clean = clean_text(text)

    # Handle empty input after cleaning
    if text_clean.strip() == "":
        return "Invalid input ❗ Please enter meaningful text."

    text_vec = vectorizer.transform([text_clean])

    prediction = model.predict(text_vec)[0]

    # Handle probability safely
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(text_vec)[0]
        confidence = prob[prediction] * 100
    else:
        confidence = 0

    if prediction == 1:
        return f"Fake Review ❌ ({confidence:.2f}%)"
    else:
        return f"Real Review ✅ ({confidence:.2f}%)"

# Streamlit UI
st.title("Fake Review Detection System")

review = st.text_area("Enter your review")

if st.button("Check Review"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        result = predict_review(review)

        if "Invalid" in result:
            st.warning(result)
        elif "Fake" in result:
            st.error(result)
        else:
            st.success(result)
