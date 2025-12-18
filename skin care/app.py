import streamlit as st
import pandas as pd
import joblib

# Load model & vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer (2).pkl")

# Load dataset
df = pd.read_csv("final_merged (2) (1).csv")

st.title("🧴 Skincare NLP System")

# Product Type Prediction
st.subheader("🔍 Predict Product Type")
user_text = st.text_area("Enter product description (after use / name / brand)")
if st.button("Predict Product Type"):
    text_vec = vectorizer.transform([user_text])
    prediction = model.predict(text_vec)
    st.success(f"Predicted Type: {prediction[0]}")

# Skin Routine Recommendation
st.subheader("💆 Skin Care Routine")
skin_type = st.selectbox("Choose your skin type:", ["Oily", "Dry", "Normal", "Combination", "Sensitive"])
def simple_skin_routine(skin_type, top_n=2):
    skin_type = skin_type.lower().capitalize()
    filtered_df = df[df[skin_type] == 1.0]
    routine_order = ['cleanser', 'toner', 'serum', 'moisturizer']
    routine = {}
    for r_type in routine_order:
        products = filtered_df[filtered_df['type'] == r_type]
        if len(products) > 0:
            routine[r_type] = products[['brand', 'type']].head(top_n)
    return routine

if st.button("Get Routine"):
    routine = simple_skin_routine(skin_type)
    for r_type, items in routine.items():
        st.markdown(f"### {r_type.upper()}")
        st.table(items)
