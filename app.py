import streamlit as st
from generator import detect_sentiment, generate_text

st.set_page_config(page_title="AI Sentiment Text Generator", layout="centered")
st.title("🧠 Sentiment-Aware AI Text Generator")

st.write("Enter a short prompt. The app detects its sentiment and generates a paragraph aligned with it.")

prompt = st.text_area("Enter your prompt:", height=150)
manual = st.checkbox("Select sentiment manually")

if manual:
    sentiment_choice = st.selectbox("Choose sentiment:", ["positive", "neutral", "negative"])
else:
    sentiment_choice = None

words = st.slider("Select word length (approx)", 50, 400, 150)
temperature = st.slider("Creativity (temperature)", 0.1, 1.2, 0.8)

if st.button("Generate Text"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        detected = detect_sentiment(prompt) if sentiment_choice is None else sentiment_choice
        with st.spinner(f"Detected sentiment: {detected}. Generating..."):
            result = generate_text(prompt, detected_sentiment=detected, words=words, temperature=temperature)
        st.success(f"Sentiment used: **{result['sentiment_used']}**")
        st.subheader("Generated Text:")
        st.write(result['generated_text'])
