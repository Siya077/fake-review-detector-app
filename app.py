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

# Your app UI
st.title("Fake Review Detection System")
