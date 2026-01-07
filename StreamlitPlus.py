import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import contractions
import pandas as pd
import os
import gdown
import zipfile

# -----------------------------
# NLTK Setup
# -----------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -----------------------------
# Helper: Download & Extract ZIP
# -----------------------------
def download_and_extract(url: str, output_folder: str):
    zip_path = output_folder + ".zip"

    if not os.path.exists(output_folder):
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_folder)
        os.remove(zip_path)

    for root, _, files in os.walk(output_folder):
        if any(f in files for f in ["config.json", "pytorch_model.bin", "model.safetensors"]):
            return root

    raise FileNotFoundError("No HuggingFace model files found.")

# -----------------------------
# Load Models (SAFE FOR STREAMLIT CLOUD)
# -----------------------------
@st.cache_resource
def load_models():
    device = torch.device("cpu")  # 🚨 FORCE CPU (Streamlit Cloud)

    mental_url = "https://drive.google.com/uc?id=1jgYUPc5ZHyzMqjK6y1mPTIEzNfVT1A-p"
    sentiment_url = "https://drive.google.com/uc?id=12Gmm6KQmY4daxf3tUDber8p3CwBu2rVV"
    label_url = "https://drive.google.com/uc?id=1njNff6TxkJEOjxAY7HU_wXnrnFzs9ulp"

    mental_dir = download_and_extract(mental_url, "saved_mental_status_bert")
    sentiment_dir = download_and_extract(sentiment_url, "saved_sentiment_model")

    # 🚫 DO NOT CALL .to(device) ON MODELS
    mental_model = AutoModelForSequenceClassification.from_pretrained(
        mental_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    )

    mental_tokenizer = AutoTokenizer.from_pretrained(mental_dir)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_dir)

    if not os.path.exists("label_encoder.pkl"):
        gdown.download(label_url, "label_encoder.pkl", quiet=False)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    mental_model.eval()
    sentiment_model.eval()

    return mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device

mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device = load_models()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = contractions.fix(text)
    text = pd.Series([text])
    text = text.str.replace(r"[^\x00-\x7F]+", "", regex=True)
    text = text.str.replace(r"@\w+|#\w+|https?://\S+|www\.\S+", "", regex=True)
    text = text.str.replace(r"\d+", "", regex=True)
    text = text.str.replace(r"[^\w\s]", "", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    text = text.apply(lambda x: " ".join(w for w in x.split() if w not in stop_words))
    return text.iloc[0]

# -----------------------------
# Labels & Suggestions
# -----------------------------
suggestions_dict = {
    "Suicidal": "Seek professional help immediately. Call a helpline or trusted person.",
    "Anxiety": "Practice mindfulness or talk to a therapist.",
    "Depression": "Maintain routines and reach out for support.",
    "Stress": "Take breaks and practice relaxation techniques.",
    "Bipolar": "Monitor moods and consult professionals.",
    "Personality disorder": "Therapy and structured routines can help.",
    "Normal": "Keep up your healthy habits."
}

label_definitions = {
    "Suicidal": "Thoughts of self-harm or hopelessness.",
    "Anxiety": "Persistent worry or fear.",
    "Depression": "Ongoing sadness and loss of interest.",
    "Stress": "Mental or physical tension.",
    "Bipolar": "Mood swings between highs and lows.",
    "Personality disorder": "Long-term behavioral patterns.",
    "Normal": "No significant concern detected."
}

# -----------------------------
# Detection Function
# -----------------------------
def detection_with_sentiment(text: str):
    cleaned = clean_text(text)

    mh_inputs = mental_tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=200
    )
    mh_inputs = {k: v.to(device) for k, v in mh_inputs.items()}

    with torch.no_grad():
        logits = mental_model(**mh_inputs).logits
        idx = torch.argmax(logits, dim=1).item()
        confidence = F.softmax(logits, dim=1)[0, idx].item()

    status = label_encoder.inverse_transform([idx])[0]
    tokens = mental_tokenizer.tokenize(cleaned)[:10]

    sent_inputs = sentiment_tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=512
    )
    sent_inputs = {k: v.to(device) for k, v in sent_inputs.items()}

    with torch.no_grad():
        probs = F.softmax(
            sentiment_model(**sent_inputs).logits, dim=-1
        ).cpu().numpy()[0]

    return {
        "status": status,
        "confidence": confidence,
        "top_words": tokens,
        "sentiment": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2]),
        },
        "suggestion": suggestions_dict[status],
        "definition": label_definitions[status],
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f8ff;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Arial Black', sans-serif;
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        font-size:16px;
        border-radius:10px;
    }
    .stTextInput>div>input {
        font-family: 'Courier New', monospace;
        font-size:16px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Mental Health & Sentiment Detection App")

input_text = st.text_input("Enter your text here:")

if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        result = detection_with_sentiment(input_text)
        status = result["status"]
        sentiment = result["sentiment"]
        confidence = result["confidence"]
        top_words = result["top_words"]
        suggestion = result["suggestion"]
        definition = result["definition"]

        # Tabs
        tabs = st.tabs(["Status", "Explanation", "Suggestions", "Sentiment"])

        # Tab 1: Status
        with tabs[0]:
            st.subheader("Mental Health Status")
            st.markdown(
                f"""
                <div style="background-color:#e0f7fa;padding:15px;border-radius:10px">
                    <h3>{status} ({confidence*100:.1f}% confidence)</h3>
                </div>
                """, unsafe_allow_html=True
            )
            with st.expander("What does this label mean?"):
                st.write(definition)

        # Tab 2: Explanation
        with tabs[1]:
            st.subheader("Why this prediction?")
            st.markdown(f"**Top contributing words:** {', '.join(top_words)}")

        # Tab 3: Suggestions
        with tabs[2]:
            st.subheader("Suggested Actions")
            st.markdown(
                f"""
                <div style="background-color:#fff3e0;padding:15px;border-radius:10px">
                    {suggestion}
                </div>
                """, unsafe_allow_html=True
            )

        # Tab 4: Sentiment
        with tabs[3]:
            st.subheader("Sentiment Analysis")
            col1, col2 = st.columns([1, 1])

            # Scores
            with col1:
                st.markdown("### Sentiment Scores")
                st.write({
                    "Negative": round(sentiment["negative"], 3),
                    "Neutral": round(sentiment["neutral"], 3),
                    "Positive": round(sentiment["positive"], 3),
                })

            # Pie chart
            with col2:
                labels = ['Negative', 'Neutral', 'Positive']
                sizes = [sentiment['negative'], sentiment['neutral'], sentiment['positive']]
                if sum(sizes) == 0:
                    sizes = [0.01,0.01,0.01]
                fig, ax = plt.subplots(figsize=(4,4))
                ax.pie(
                    sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                    explode=(0.05,0.05,0.05), shadow=True
                )
                ax.set_title('Sentiment Distribution')
                st.pyplot(fig)
