import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
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
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# Helper: Download and extract ZIP
# -----------------------------
def download_and_extract(url: str, output_folder: str):
    """
    Downloads a ZIP file from Google Drive and extracts it to a folder.
    Automatically detects the inner folder containing model files.
    """
    zip_path = output_folder + ".zip"

    # Ensure parent folder exists
    parent_dir = os.path.dirname(output_folder)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Download and extract only if folder doesn't exist
    if not os.path.exists(output_folder):
        print(f"Downloading {output_folder}...")
        gdown.download(url, zip_path, quiet=False)
        print(f"Extracting {output_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        os.remove(zip_path)
        print(f"{output_folder} downloaded and extracted.")

    # Detect folder containing HuggingFace model files
    for root, dirs, files in os.walk(output_folder):
        if any(f in files for f in ["config.json", "pytorch_model.bin", "model.safetensors"]):
            return root  # This folder can be loaded by HuggingFace

    raise FileNotFoundError(f"No HuggingFace model files found in {output_folder}")

# -----------------------------
# Load Models
# -----------------------------
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Google Drive URLs
    mental_url = "https://drive.google.com/uc?id=1jgYUPc5ZHyzMqjK6y1mPTIEzNfVT1A-p"
    sentiment_url = "https://drive.google.com/uc?id=12Gmm6KQmY4daxf3tUDber8p3CwBu2rVV"
    label_url = "https://drive.google.com/uc?id=1njNff6TxkJEOjxAY7HU_wXnrnFzs9ulp"

    # Download & extract models
    mental_inner = download_and_extract(mental_url, "saved_mental_status_bert")
    sentiment_inner = download_and_extract(sentiment_url, "saved_sentiment_model")

    # Load HuggingFace models
    print("Loading mental health model...")
    mental_model = AutoModelForSequenceClassification.from_pretrained(mental_inner).to(device)
    mental_tokenizer = AutoTokenizer.from_pretrained(mental_inner)

    print("Loading sentiment model...")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_inner).to(device)
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_inner)

    # Load label encoder
    if not os.path.exists("label_encoder.pkl"):
        print("Downloading label encoder...")
        gdown.download(label_url, "label_encoder.pkl", quiet=False)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Set models to eval mode
    mental_model.eval()
    sentiment_model.eval()

    print("Models and tokenizers are ready.")
    return mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device

# Load models
mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device = load_models()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    is_string = False
    if isinstance(text, str):
        text = pd.Series([text])
        is_string = True

    text = text.str.lower()
    text = text.apply(contractions.fix)
    text = text.str.replace(r'[^\x00-\x7F]+', '', regex=True)
    text = text.str.replace(r'@\w+|#\w+|https?://\S+|www\.\S+|<.*?>|\[.*?\]', '', regex=True)
    text = text.str.replace(r'\d+', '', regex=True)
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace('-', ' ', regex=False)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    if is_string:
        return text.iloc[0]
    return text

# -----------------------------
# Suggestions & Labels
# -----------------------------
suggestions_dict = {
    "Suicidal": "Seek professional help immediately. Call helplines or talk to a trusted person.",
    "Anxiety": "Practice deep breathing, mindfulness, or talk to a therapist.",
    "Depression": "Engage in daily routines, physical activity, and reach out for support.",
    "Stress": "Take breaks, practice relaxation techniques, and prioritize tasks.",
    "Bipolar": "Monitor moods, adhere to treatment, and consult mental health professionals.",
    "Personality disorder": "Therapy and self-awareness can help manage symptoms. Build supportive routines.",
    "Normal": "Keep up your healthy habits."
}

label_definitions = {
    "Suicidal": "Intense feelings of hopelessness and thoughts of self-harm.",
    "Anxiety": "Excessive worry or fear that affects daily activities.",
    "Depression": "Persistent sadness, lack of energy, and loss of interest.",
    "Stress": "Physical or mental tension due to challenging situations.",
    "Bipolar": "Mood disorder with alternating manic and depressive episodes.",
    "Personality disorder": "Enduring patterns of behavior affecting social and emotional life.",
    "Normal": "No significant mental health concern detected."
}

# -----------------------------
# Detection Function
# -----------------------------
def detection_with_sentiment(text):
    cleaned_text = clean_text(text)

    # Mental health detection
    mh_inputs = mental_tokenizer(cleaned_text, padding=True, truncation=True, return_tensors="pt", max_length=200)
    mh_inputs = {k: v.to(device) for k, v in mh_inputs.items()}
    with torch.no_grad():
        mh_outputs = mental_model(**mh_inputs)
        mh_logits = mh_outputs.logits
        predicted_label_idx = torch.argmax(mh_logits, dim=1).cpu().item()
        status = label_encoder.inverse_transform([predicted_label_idx])[0]
        confidence = F.softmax(mh_logits, dim=1)[0, predicted_label_idx].cpu().item()
    tokens = mental_tokenizer.tokenize(cleaned_text)
    top_words = tokens[:10]

    # Sentiment detection
    sent_inputs = sentiment_tokenizer(cleaned_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    sent_inputs = {k: v.to(device) for k, v in sent_inputs.items()}
    with torch.no_grad():
        sent_outputs = sentiment_model(**sent_inputs)
        probs = F.softmax(sent_outputs.logits, dim=-1).cpu().numpy()[0]
        sentiment = {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2])
        }

    return {
        "status": status,
        "confidence": confidence,
        "top_words": top_words,
        "sentiment": sentiment,
        "suggestion": suggestions_dict.get(status, "No suggestion available."),
        "definition": label_definitions.get(status, "")
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #f0f8ff; }
h1, h2, h3, h4 { font-family: 'Arial Black', sans-serif; color: #1e3d59; }
.stButton>button { background-color: #ff6f61; color: white; font-size:16px; border-radius:10px; }
.stTextInput>div>input { font-family: 'Courier New', monospace; font-size:16px; }
</style>
""", unsafe_allow_html=True)

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

        tabs = st.tabs(["Status", "Explanation", "Suggestions", "Sentiment"])

        with tabs[0]:
            st.subheader("Mental Health Status")
            st.markdown(f"<div style='background-color:#e0f7fa;padding:15px;border-radius:10px'><h3>{status} ({confidence*100:.1f}% confidence)</h3></div>", unsafe_allow_html=True)
            with st.expander("What does this label mean?"):
                st.write(definition)

        with tabs[1]:
            st.subheader("Why this prediction?")
            st.markdown(f"**Top contributing words:** {', '.join(top_words)}")

        with tabs[2]:
            st.subheader("Suggested Actions")
            st.markdown(f"<div style='background-color:#fff3e0;padding:15px;border-radius:10px'>{suggestion}</div>", unsafe_allow_html=True)

        with tabs[3]:
            st.subheader("Sentiment Analysis")
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("### Sentiment Scores")
                st.write({ "Negative": round(sentiment["negative"], 3), "Neutral": round(sentiment["neutral"], 3), "Positive": round(sentiment["positive"], 3) })
            with col2:
                labels = ['Negative', 'Neutral', 'Positive']
                sizes = [sentiment['negative'], sentiment['neutral'], sentiment['positive']]
                if sum(sizes) == 0: sizes = [0.01,0.01,0.01]
                fig, ax = plt.subplots(figsize=(4,4))
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=(0.05,0.05,0.05), shadow=True)
                ax.set_title('Sentiment Distribution')
                st.pyplot(fig)
