import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
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

# SETUP
nltk.data.path.append("./nltk_data")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def download_and_extract(url: str, output_folder: str):
    zip_path = output_folder + ".zip"

    # Ensure folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Download zip if not exists
    if not os.path.exists(zip_path):
        print(f"Downloading {output_folder}...")
        gdown.download(url, zip_path, quiet=False)

    # Extract
    print(f"Extracting {output_folder}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    
    # Remove zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Detect inner folder containing HuggingFace model files
    for root, dirs, files in os.walk(output_folder):
        if "config.json" in files or "pytorch_model.bin" in files or "model.safetensors" in files:
            return root

    raise FileNotFoundError(f"No HuggingFace model files found in {output_folder}")

# LOAD MODELS
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Google Drive URLs
    mental_url = "https://drive.google.com/uc?id=1jgYUPc5ZHyzMqjK6y1mPTIEzNfVT1A-p"
    sentiment_url = "https://drive.google.com/uc?id=12Gmm6KQmY4daxf3tUDber8p3CwBu2rVV"
    label_url = "https://drive.google.com/uc?id=1njNff6TxkJEOjxAY7HU_wXnrnFzs9ulp"

    # Download & extract
    mental_inner = download_and_extract(mental_url, "saved_mental_status_bert")
    sentiment_inner = download_and_extract(sentiment_url, "saved_sentiment_model")

    # Load HuggingFace models safely
    print("Loading mental health model...")
    mental_model = AutoModelForSequenceClassification.from_pretrained(mental_inner, device_map="auto")
    mental_tokenizer = AutoTokenizer.from_pretrained(mental_inner)

    print("Loading sentiment model...")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_inner, device_map="auto")
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_inner)

    # Load label encoder
    if not os.path.exists("label_encoder.pkl"):
        print("Downloading label encoder...")
        gdown.download(label_url, "label_encoder.pkl", quiet=False)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    mental_model.eval()
    sentiment_model.eval()

    print("Models and tokenizers are ready.")
    return mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device

# Initialize models
mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device = load_models()

# TEXT CLEANING FUNCTION
def clean_text(text):

    # Check if text is a string
    is_string = False
    if isinstance(text, str):
        text = pd.Series([text])
        is_string = True

    # Convert to lowercase first
    text = text.str.lower()

    # Expand contractions (I've → I have)
    text = text.apply(contractions.fix)

    # Remove emojis (non-ASCII characters)
    text= text.str.replace(r'[^\x00-\x7F]+', '', regex=True)

    # Remove @mentions, hashtags, URLs, HTML, and brackets
    text = text.str.replace(r'@\w+', '', regex=True)
    text = text.str.replace(r'#\w+', '', regex=True)
    text = text.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
    text= text.str.replace(r'<.*?>', '', regex=True)
    text = text.str.replace(r'\[.*?\]', '', regex=True)

    # Remove digits and punctuation
    text= text.str.replace(r'\d+', '', regex=True)
    text = text.str.replace(r'[^\w\s]', '', regex=True)

    # Replace hyphens with spaces
    text = text.str.replace('-', ' ', regex=False)

    # Remove newline characters and extra spaces
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()

    # Remove stopwords
    text = text.apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    # If original input was a string, return a string
    if is_string:
        return text.iloc[0]

    return text

# Suggestion based on status
suggestions_dict = {
    "Suicidal": "Seek professional help immediately. Call helplines or talk to a trusted person.",
    "Anxiety": "Practice deep breathing, mindfulness, or talk to a therapist.",
    "Depression": "Engage in daily routines, physical activity, and reach out for support.",
    "Stress": "Take breaks, practice relaxation techniques, and prioritize tasks.",
    "Bipolar": "Monitor moods, adhere to treatment, and consult mental health professionals.",
    "Personality disorder": "Therapy and self-awareness can help manage symptoms. Build supportive routines.",
    "Normal": "Keep up your healthy habits."
}

# Label definitions
label_definitions = {
    "Suicidal": "Intense feelings of hopelessness and thoughts of self-harm.",
    "Anxiety": "Excessive worry or fear that affects daily activities.",
    "Depression": "Persistent sadness, lack of energy, and loss of interest.",
    "Stress": "Physical or mental tension due to challenging situations.",
    "Bipolar": "Mood disorder with alternating manic and depressive episodes.",
    "Personality disorder": "Enduring patterns of behavior affecting social and emotional life.",
    "Normal": "No significant mental health concern detected."
}


# DETECTION FUNCTION (status + sentiment)
def detection_with_sentiment(text):
    cleaned_text = clean_text(text)

    # Mental health detection
    mh_inputs = mental_tokenizer(cleaned_text, padding=True, truncation=True, return_tensors="pt", max_length=200)
    mh_inputs = {k: v.to(device) for k, v in mh_inputs.items()}
    
    with torch.no_grad():
        mh_outputs = mental_model(**mh_inputs)
        mh_logits = mh_outputs.logits
        predicted_label_idx = torch.argmax(mh_logits, dim=1).item()
        status = label_encoder.inverse_transform([predicted_label_idx])[0]
        confidence = F.softmax(mh_logits, dim=1)[0, predicted_label_idx].item()

     # Simple explainability: top contributing words via token embeddings
    tokens = mental_tokenizer.tokenize(cleaned_text)
    token_ids = mental_tokenizer.convert_tokens_to_ids(tokens)
    token_logits = mh_logits[0][predicted_label_idx].item()
    top_words = tokens[:10] 
    #Sentiment detection 
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

def explain_prediction(text, model, tokenizer, target_label):
    tokens = tokenizer.tokenize(text)
    explanations = []

    for token in set(tokens):
        masked_text = text.replace(token, "[MASK]")
        inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=200).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]

        explanations.append((token, probs[target_label].item()))

    explanations = sorted(explanations, key=lambda x: x[1], reverse=True)
    return explanations[:5]



# STREAMLIT UI
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



# TEST EXAMPLES 
# Example Sentences
# 
# - **Suicidal:** I feel hopeless and tired all the time. Nothing seems to matter anymore.
# - **Normal:** I had breakfast, went to work and came home. It is an ordinary day.
# - **Anxiety:** My heart races and I can’t stop worrying about everything that might go wrong.
# - **Depression:** I feel hopeless, sad, and I have no energy to do anything.
# - **Stress:** I am overwhelmed with work and feel very stressed.
# - **Bipolar:** I am having a manic episode.
# - **Personality disorder:**I think I have AvPD as I always try to avoid myself from social interaction.