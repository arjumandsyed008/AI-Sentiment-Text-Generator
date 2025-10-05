# generator.py
from transformers import pipeline, set_seed

# Load pretrained sentiment analysis model (Twitter sentiment)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
)

# Load GPT-2 for text generation
generator = pipeline("text-generation", model="gpt2")
set_seed(42)


def detect_sentiment(text):
    """
    Detect sentiment (positive, negative, neutral) from input text.
    """
    try:
        result = sentiment_pipe(text)
        label = result[0]['label'].lower()
        # Some models return "LABEL_0" etc., map them properly:
        mapping = {
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive"
        }
        return mapping.get(label, "neutral")
    except Exception as e:
        print("Sentiment detection error:", e)
        return "neutral"


def make_generation_prompt(user_prompt, sentiment_label):
    """
    Create a generation prompt tailored to sentiment.
    """
    sentiment_prompts = {
        "positive": "Write a positive and uplifting paragraph about:",
        "negative": "Write a critical and negative paragraph about:",
        "neutral": "Write a neutral and balanced paragraph about:"
    }
    prefix = sentiment_prompts.get(sentiment_label, "Write a balanced paragraph about:")
    return f"{prefix} {user_prompt}\n\nParagraph:"


def generate_text(user_prompt, detected_sentiment=None, words=150, do_sample=True, temperature=0.8):
    """
    Generate text using GPT-2 with sentiment alignment.
    """
    try:
        if detected_sentiment is None:
            detected_sentiment = detect_sentiment(user_prompt)

        prompt = make_generation_prompt(user_prompt, detected_sentiment)
        max_length = min(1024, int(words * 1.6) + 50)

        output = generator(
            prompt,
            max_length=max_length,
            do_sample=do_sample,
            top_k=50,
            top_p=0.9,
            temperature=temperature,
            num_return_sequences=1,
        )

        # Extract generated portion only
        full_text = output[0]['generated_text']
        generated_text = full_text.replace(prompt, "").strip()

        return {
            "sentiment_used": detected_sentiment,
            "generated_text": generated_text
        }

    except Exception as e:
        print("Text generation error:", e)
        return {
            "sentiment_used": detected_sentiment or "neutral",
            "generated_text": "⚠️ Error during generation. Check console for details."
        }
