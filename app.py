import streamlit as st
import torch
from transformers import pipeline

@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model="hamzab/roberta-fake-news-classification",
        tokenizer="hamzab/roberta-fake-news-classification"
    )
    return classifier

classifier = load_model()

st.set_page_config(page_title="AI Fake News Detector", page_icon="📰")

st.title("📰 AI-Based Fake News Detection")
st.markdown("**Powered by RoBERTa**")
st.caption("⚠️ Note: This is an AI model and may not be 100% accurate. Use for educational/demo purposes only.")
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a fine-tuned RoBERTa model to classify news as REAL or FAKE.")
    st.write("Model: hamzab/roberta-fake-news-classification")

news_text = st.text_area(
    "📝 Paste News Content Here",
    height=200,
    placeholder="Type or paste any English news here..."
)

col1, col2 = st.columns([3, 1])
with col1:
    detect_clicked = st.button("🔍 Detect News", use_container_width=True)
with col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if detect_clicked:
    text = news_text.strip()

    if text == "":
        st.warning("⚠️ Please enter some news text!")
    elif len(text.split()) < 10:
        st.warning("⚠️ Please enter a longer news article (at least 10-15 words) for accurate detection.")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(text[:512])[0]
            raw_label = result['label'].upper()
            confidence = result['score'] * 100

            if raw_label in ["LABEL_1", "REAL", "TRUE"]:
                prediction = "REAL"
            else:
                prediction = "FAKE"

        st.markdown("---")

        if confidence < 70:
            st.info(f"🤔 Result Uncertain (Confidence: {confidence:.2f}%) — try a more detailed article.")
        else:
            if prediction == "REAL":
                st.success("✅ This News is **REAL**")
            else:
                st.error("🚫 This News is **FAKE**")
            st.info(f"🎯 Confidence Score: {confidence:.2f}%")