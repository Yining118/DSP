import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MarianMTModel, MarianTokenizer
import pickle
import nltk
from nltk.corpus import stopwords
import contractions
import pandas as pd
import os
import gdown
import zipfile
import random
import praw
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import MarianMTModel, MarianTokenizer

import os
hf_token = os.environ.get("HF_TOKEN")

# -------------------------------
# NLTK Setup
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Load Models & Tokenizers
# -------------------------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mental_model = AutoModelForSequenceClassification.from_pretrained(
        "saved_mental_status_bert", device_map="cpu"
    )
    mental_model.to(device)

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        "saved_sentiment_model", device_map="cpu"
    )
    sentiment_model.to(device)

    mental_tokenizer = AutoTokenizer.from_pretrained("saved_mental_status_bert", use_fast=False)
    sentiment_tokenizer = AutoTokenizer.from_pretrained("saved_sentiment_model", use_fast=False)
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    return mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device

mental_model, mental_tokenizer, sentiment_model, sentiment_tokenizer, label_encoder, device = load_models()

# -------------------------------
# Clean Text Function
# -------------------------------
def clean_text(text):
    is_string = False
    if isinstance(text, str):
        text = pd.Series([text])
        is_string = True

    text = text.str.lower()
    text = text.apply(contractions.fix)
    text = text.str.replace(r'[^\x00-\x7F]+', '', regex=True)
    text = text.str.replace(r'@\w+', '', regex=True)
    text = text.str.replace(r'#\w+', '', regex=True)
    text = text.str.replace(r'https?://\S+|www\.\S+', '', regex=True)
    text = text.str.replace(r'<.*?>', '', regex=True)
    text = text.str.replace(r'\[.*?\]', '', regex=True)
    text = text.str.replace(r'\d+', '', regex=True)
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace('-', ' ', regex=False)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    text = text.apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
    if is_string:
        return text.iloc[0]
    return text

# -------------------------------
# Suggestions & Definitions
# -------------------------------
suggestions_dict = {
    "Suicidal": "Seek professional help immediately. Call helplines or talk to a trusted person.",
    "Anxiety": "Practice deep breathing, mindfulness, or talk to a therapist.",
    "Depression": "Engage in daily routines, physical activity, and reach out for support.",
    "Stress": "Take breaks, practice relaxation techniques, and prioritize tasks.",
    "Bipolar": "Monitor moods, adhere to treatment, and consult mental health professionals.",
    "Personality disorder": "Therapy and self-awareness can help manage symptoms. Build supportive routines.",
    "Normal": "Keep up your healthy habits."
}

suggestions_dict_malay = {
    "Suicidal": "Dapatkan bantuan profesional segera. Hubungi talian bantuan atau bercakap dengan orang yang dipercayai.",
    "Anxiety": "Amalkan pernafasan dalam, kesedaran minda, atau bercakap dengan kaunselor.",
    "Depression": "Teruskan rutin harian, aktiviti fizikal, dan dapatkan sokongan.",
    "Stress": "Berehat, amalkan teknik relaksasi, dan utamakan tugas.",
    "Bipolar": "Pantau mood, patuhi rawatan, dan rujuk profesional kesihatan mental.",
    "Personality disorder": "Terapi dan kesedaran diri boleh membantu mengurus simptom. Wujudkan rutin sokongan.",
    "Normal": "Teruskan tabiat sihat anda."
}

label_names_malay = {
    "Suicidal": "Cenderung membunuh diri",
    "Anxiety": "Kebimbangan",
    "Depression": "Kemurungan",
    "Stress": "Tekanan",
    "Bipolar": "Bipolar",
    "Personality disorder": "Gangguan personaliti",
    "Normal": "Normal"
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

label_definitions_malay = {
    "Suicidal": "Perasaan putus asa yang teruk dan pemikiran untuk mencederakan diri sendiri.",
    "Anxiety": "Kebimbangan berlebihan yang menjejaskan aktiviti harian.",
    "Depression": "Kesedihan berterusan, kehilangan tenaga, dan hilang minat.",
    "Stress": "Ketegangan fizikal atau mental akibat situasi mencabar.",
    "Bipolar": "Gangguan mood dengan episod manik dan kemurungan bergilir-gilir.",
    "Personality disorder": "Corak tingkah laku berterusan yang menjejaskan kehidupan sosial dan emosi.",
    "Normal": "Tiada kebimbangan kesihatan mental yang signifikan dikesan."
}

awareness_info= {
    "Suicidal": {
        "Summary": {
            "English": "Intense feelings of hopelessness and thoughts of self-harm.",
            "Malay": "Perasaan putus asa yang teruk dan pemikiran untuk mencederakan diri sendiri."
        },
        "Actions": {
            "English": [
                "📞 Seek professional help immediately (Malaysia: Befrienders 03-7956 8145).",
                "🗣️ Talk to a trusted person or family member.",
                "🌐 Visit Malaysian mental health resources: www.befrienders.org.my"
            ],
            "Malay": [
                "📞 Dapatkan bantuan profesional segera (Malaysia: Befrienders 03-7956 8145).",
                "🗣️ Bercakap dengan orang yang dipercayai atau ahli keluarga.",
                "🌐 Lawati sumber kesihatan mental Malaysia: www.befrienders.org.my"
            ]
        }
    },
    "Anxiety": {
        "Summary": {
            "English": "Excessive worry or fear that affects daily activities.",
            "Malay": "Kebimbangan berlebihan yang menjejaskan aktiviti harian."
        },
        "Actions": {
            "English": [
                "🧘 Practice deep breathing or mindfulness.",
                "🗣️ Consult a therapist in Malaysia.",
                "🌐 Check online resources from Malaysian mental health organizations."
            ],
            "Malay": [
                "🧘 Amalkan pernafasan dalam atau kesedaran minda.",
                "🗣️ Rujuk kaunselor atau terapi di Malaysia.",
                "🌐 Semak sumber dalam talian daripada organisasi kesihatan mental Malaysia."
            ]
        }
    },
    "Depression": {
        "Summary": {
            "English": "Persistent sadness, lack of energy, and loss of interest.",
            "Malay": "Kesedihan berterusan, kehilangan tenaga, dan hilang minat."
        },
        "Actions": {
            "English": [
                "🏃 Engage in daily routines and physical activity.",
                "🧑‍⚕️ Seek professional support in Malaysia.",
                "💬 Talk to friends or family."
            ],
            "Malay": [
                "🏃 Teruskan rutin harian dan aktiviti fizikal.",
                "🧑‍⚕️ Dapatkan sokongan profesional di Malaysia.",
                "💬 Bercakap dengan rakan atau keluarga."
            ]
        }
    },
    "Stress": {
        "Summary": {
            "English": "Physical or mental tension due to challenging situations.",
            "Malay": "Ketegangan fizikal atau mental akibat situasi mencabar."
        },
        "Actions": {
            "English": [
                "🛀 Take breaks and practice relaxation techniques.",
                "📝 Prioritize tasks and manage time.",
                "🌐 Explore Malaysian stress management workshops."
            ],
            "Malay": [
                "🛀 Berehat dan amalkan teknik relaksasi.",
                "📝 Utamakan tugas dan urus masa.",
                "🌐 Sertai bengkel pengurusan tekanan di Malaysia."
            ]
        }
    },
    "Bipolar": {
        "Summary": {
            "English": "Mood disorder with alternating manic and depressive episodes.",
            "Malay": "Gangguan mood dengan episod manik dan kemurungan bergilir-gilir."
        },
        "Actions": {
            "English": [
                "📝 Monitor moods regularly.",
                "💊 Follow treatment plans.",
                "🧑‍⚕️ Consult Malaysian mental health professionals."
            ],
            "Malay": [
                "📝 Pantau mood secara berkala.",
                "💊 Patuh pada pelan rawatan.",
                "🧑‍⚕️ Rujuk profesional kesihatan mental di Malaysia."
            ]
        }
    },
    "Personality disorder": {
        "Summary": {
            "English": "Enduring patterns of behavior affecting social and emotional life.",
            "Malay": "Corak tingkah laku berterusan yang menjejaskan kehidupan sosial dan emosi."
        },
        "Actions": {
            "English": [
                "🧠 Consider therapy and self-awareness exercises.",
                "🤝 Build supportive routines and relationships.",
                "🌐 Use Malaysian mental health resources."
            ],
            "Malay": [
                "🧠 Pertimbangkan terapi dan latihan kesedaran diri.",
                "🤝 Wujudkan rutin sokongan dan hubungan.",
                "🌐 Gunakan sumber kesihatan mental Malaysia."
            ]
        }
    },
    "Normal": {
        "Summary": {
            "English": "No significant mental health concern detected.",
            "Malay": "Tiada isu kesihatan mental yang signifikan dikesan."
        },
        "Actions": {
            "English": [
                "✅ Maintain healthy habits and regular checkups.",
                "🧘 Continue stress management and mindfulness."
            ],
            "Malay": [
                "✅ Teruskan tabiat sihat dan pemeriksaan berkala.",
                "🧘 Teruskan pengurusan tekanan dan kesedaran minda."
            ]
        }
    }
}


# -------------------------------
# Load MarianMT Translation Model
# -------------------------------
@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = MarianTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = MarianMTModel.from_pretrained(model_name, use_auth_token=hf_token)
    model.eval()
    return tokenizer, model

translation_tokenizer, translation_model = load_translation_model()

def translate_malay_to_english(text):
    if not text.strip():
        return ""
    inputs = translation_tokenizer(text, return_tensors="pt", truncation=True)
    translated = translation_model.generate(**inputs)
    return translation_tokenizer.decode(translated[0], skip_special_tokens=True)

# -------------------------------
# Detection Function
# -------------------------------
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

    # Top words
    tokens = mental_tokenizer.tokenize(cleaned_text)
    top_words = tokens[:100]

    # Sentiment detection
    sent_inputs = sentiment_tokenizer(cleaned_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    sent_inputs = {k: v.to(device) for k, v in sent_inputs.items()}
    with torch.no_grad():
        sent_outputs = sentiment_model(**sent_inputs)
        probs = F.softmax(sent_outputs.logits, dim=-1).cpu().numpy()[0]
        sentiment = {"negative": float(probs[0]), "neutral": float(probs[1]), "positive": float(probs[2])}

    return {
        "status": status,
        "confidence": confidence,
        "top_words": top_words,
        "sentiment": sentiment
    }

# -------------------------------
# Reddit Fetch
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_reddit_posts(keyword, limit=100):
    reddit = praw.Reddit(
        client_id="eAKZnHy0bMWsFzfoLkkVcA",
        client_secret="MwzVtX2XnFPCGRjJs5NZZHDTrkwUkA",
        user_agent="MentalHealthAnalysis:v1.0"
    )

    posts = []
    for submission in reddit.subreddit("all").search(keyword, sort='new', limit=limit):
        if submission.selftext and len(submission.selftext) > 20:
            posts.append(submission.selftext)
    return posts

# -------------------------------
# Streamlit UI
# -------------------------------
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Raleway:wght@400;600;700&display=swap');

/* Background image */
.stApp {
    background-image: 
        linear-gradient(rgba(240,248,255,0.85), rgba(208,231,255,0.85)),
        url("https://tse1.mm.bing.net/th/id/OIP.1kxFfC7PVvOtdc8Uoz_uLwHaEK?rs=1&pid=ImgDetMain&o=7&rm=3");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Raleway', sans-serif;
    color: #1e3d59;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

/* Buttons */
.stButton>button {
    background-color: #ff6f61;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 15px;
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
}
.stButton>button:hover {
    background-color: #e55b50;
    transform: translateY(-2px);
    box-shadow: 0px 6px 10px rgba(0,0,0,0.2);
}

/* Text input and textarea */
.stTextInput>div>input, .stTextArea>div>textarea {
    font-family: 'Poppins', sans-serif;
    font-size: 16px;
    border-radius: 10px;
    padding: 8px;
    border: 1px solid #ccc;
}

/* Tabs styling */
.css-1n76uvr.edgvbvh3 {
    font-family: 'Raleway', sans-serif;
    font-weight: 600;
    font-size: 16px;
}

/* Expander styling */
.stExpanderHeader {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    color: #1e3d59;
}

/* Markdown text inside cards */
div[style*="background-color:#e0f7fa"], div[style*="background-color:#fff3e0"] {
    font-family: 'Poppins', sans-serif;
    font-size: 15px;
}

/* Pie chart labels */
svg text {
    font-family: 'Raleway', sans-serif;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Language selector
language = st.selectbox("Choose language / Pilih bahasa:", ["English", "Malay"])

# Title
st.title(
    "🌟 Mental Health & Sentiment Detection App 🌟"
    if language == "English"
    else "🌟 Aplikasi Pengesanan Kesihatan Mental & Sentimen 🌟"
)

# Input
input_text = st.text_area(
    "Enter your text (Malay or English):"
    if language == "English"
    else "Masukkan teks anda (Bahasa Melayu atau Inggeris):"
)

# Translate Malay → English
translated_text = translate_malay_to_english(input_text) if language=="Malay" else input_text
if translated_text:
    st.write("Translated Text:" if language=="English" else "Teks Terjemahan:", translated_text)

# Initialize session state
if "result" not in st.session_state:
    st.session_state["result"] = None
if "social_result" not in st.session_state:
    st.session_state["social_result"] = None
if "show_awareness" not in st.session_state:
    st.session_state["show_awareness"] = False

# Detect Button
if st.button("Detect" if language=="English" else "Kesan"):
    if not translated_text.strip():
        st.warning("Please enter some text." if language=="English" else "Sila masukkan teks.")
    else:
        st.session_state["result"] = detection_with_sentiment(translated_text)


# Only proceed if detection has been done
if st.session_state["result"]:
    result = st.session_state["result"]
    status = result["status"]
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    top_words = result["top_words"]

    # Get language-specific display
    display_label = status if language == "English" else label_names_malay.get(status, status)
    display_definition = (
        label_definitions[status]
        if language == "English"
        else label_definitions_malay.get(status, label_definitions[status])
    )

    # Tabs
    tabs = st.tabs(
        ["Status", "Explanation", "Suggestions", "Sentiment", "Social Media"]
        if language == "English"
        else ["Status", "Penerangan", "Cadangan", "Sentimen", "Media Sosial"]
    )

    # Tab 1: Status
    with tabs[0]:
        st.subheader(
            "Mental Health Status" if language == "English" else "Status Kesihatan Mental"
        )
        st.markdown(
            f"<div style='background-color:#e0f7fa;padding:15px;border-radius:10px'>"
            f"<h3>{display_label} ({confidence*100:.1f}% {'confidence' if language=='English' else 'keyakinan'})</h3></div>",
            unsafe_allow_html=True,
        )

        with st.expander("What does this label mean?" if language == "English" else "Apa maksud label ini?"):
            st.write(display_definition)

    # Tab 2: Explanation
    with tabs[1]:
        st.subheader("Top Contributing Words" if language == "English" else "Perkataan Penyumbang Utama")
        if language == "Malay":
            try:
                top_text = " ".join(top_words)
                top_text_malay = translator.translate(top_text, src="en", dest="ms").text
                top_words_display = top_text_malay.split()
            except:
                top_words_display = top_words
        else:
            top_words_display = top_words
        st.markdown(f"**{', '.join(top_words_display)}**")

    # Tab 3: Suggestions
    with tabs[2]:
        st.subheader("Suggested Actions" if language == "English" else "Cadangan Tindakan")
        display_suggestion = (
            suggestions_dict.get(status)
            if language == "English"
            else suggestions_dict_malay.get(status)
        )
        st.markdown(
            f"<div style='background-color:#fff3e0;padding:15px;border-radius:10px'>{display_suggestion}</div>",
            unsafe_allow_html=True,
        )

        # Awareness Button
        if st.button("Find Out More About Mental Health" if language == "English" else "Ketahui Lebih Lanjut Mengenai Kesihatan Mental"):
            st.session_state["show_awareness"] = True

        if st.session_state["show_awareness"]:
            st.subheader("Mental Health Awareness in Malaysia" if language == "English" else "Kesedaran Kesihatan Mental di Malaysia")
            for label, info in awareness_info.items():
                with st.expander(label if language == "English" else label_names_malay[label]):
                    st.markdown(f"**{info['Summary'][language]}**")
                    st.write("Actions / Tindakan:" if language == "English" else "Tindakan:")
                    for action in info["Actions"][language]:
                        st.write(f"- {action}")

    # Tab 4: Sentiment
    with tabs[3]:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                f"""
                <div style="background-color: rgba(255,255,255,0); padding:10px; border-radius:10px;">
                    <p>{'Negatif' if language=='Malay' else 'Negative'}: {round(sentiment.get('negative',0),3)}</p>
                    <p>{'Neutral' if language=='Malay' else 'Neutral'}: {round(sentiment.get('neutral',0),3)}</p>
                    <p>{'Positif' if language=='Malay' else 'Positive'}: {round(sentiment.get('positive',0),3)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Flip card for mood
            dominant = max(sentiment, key=sentiment.get)
            tips = {
                "negative": [
                    "Take a deep breath and smile 😊",
                    "Listen to your favorite music today!",
                    "Try a short walk to refresh your mind.",
                ]
                if language == "English"
                else [
                    "Tarik nafas dalam dan senyum 😊",
                    "Dengar muzik kegemaran anda hari ini!",
                    "Cuba berjalan sebentar untuk menyegarkan minda.",
                ],
                "neutral": [
                    "Keep doing what you’re doing!",
                    "Take a short break and enjoy a snack.",
                    "Balance your day with some fun activities.",
                ]
                if language == "English"
                else [
                    "Teruskan aktiviti anda!",
                    "Berehat sebentar dan nikmati snek.",
                    "Seimbangkan hari anda dengan aktiviti menyeronokkan.",
                ],
                "positive": [
                    "Keep up the great mood! 🌟",
                    "Share your happiness with someone today.",
                    "Celebrate the little wins!",
                ]
                if language == "English"
                else [
                    "Teruskan mood baik! 🌟",
                    "Kongsi kebahagiaan anda dengan seseorang.",
                    "Raikan kejayaan kecil!",
                ],
            }
            mood_messages = {
                "negative": "You are in a bad mood 😔" if language == "English" else "Anda berada dalam mood buruk 😔",
                "neutral": "You are feeling okay 🙂" if language == "English" else "Anda rasa biasa-biasa sahaja 🙂",
                "positive": "You are in a good mood 😄" if language == "English" else "Anda berada dalam mood baik 😄",
            }
            mood_message = mood_messages[dominant.lower()]
            tip_message = random.choice(tips[dominant.lower()])

            flip_card_html = f"""
            <style>
            .flip-card {{
                background-color: transparent;
                width: 300px;
                height: 150px;
                perspective: 1000px;
                margin: 15px auto;
            }}
            .flip-card-inner {{
                position: relative;
                width: 100%;
                height: 100%;
                text-align: center;
                transition: transform 0.8s;
                transform-style: preserve-3d;
            }}
            .flip-card:hover .flip-card-inner {{
                transform: rotateY(180deg);
            }}
            .flip-card-front, .flip-card-back {{
                position: absolute;
                width: 100%;
                height: 100%;
                -webkit-backface-visibility: hidden;
                backface-visibility: hidden;
                border-radius: 15px;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 18px;
                padding: 10px;
                color: white;
            }}
            .flip-card-front {{
                background-color: #ff6f61;
            }}
            .flip-card-back {{
                background-color: #1e3d59;
                transform: rotateY(180deg);
            }}
            </style>
            <div class="flip-card">
                <div class="flip-card-inner">
                    <div class="flip-card-front">{mood_message}</div>
                    <div class="flip-card-back">{tip_message}</div>
                </div>
            </div>
            """
            st.markdown(flip_card_html, unsafe_allow_html=True)

        with col2:
            labels = ['Negative','Neutral','Positive'] if language=="English" else ['Negatif','Neutral','Positif']
            sizes = [sentiment.get('negative',0), sentiment.get('neutral',0), sentiment.get('positive',0)]
            fig, ax = plt.subplots(figsize=(4,4))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.pie(
                sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                explode=(0.05,0.05,0.05), shadow=True
            )
            st.pyplot(fig, transparent=True)

    # Tab 5: Social Media Analysis
    with tabs[4]:
        st.subheader(
            "Near Real-Time Social Media Mental Health Analysis"
            if language == "English"
            else "Analisis Media Sosial Hampir Masa Nyata"
        )

        keyword = st.text_input(
            "Enter keyword to analyze (e.g. stress, exam, burnout)"
            if language == "English"
            else "Masukkan kata kunci (contoh: stres, peperiksaan)"
        )

        num_posts = st.slider(
            "Number of posts"
            if language == "English"
            else "Bilangan hantaran",
            min_value=50,
            max_value=300,
            step=50,
            value=100
        )

        # Initialize session state for posts
        if "social_posts" not in st.session_state:
            st.session_state["social_posts"] = []

        # Analyze button
        if st.button("Analyze Social Media" if language == "English" else "Analisis Media Sosial"):
            if not keyword.strip():
                st.warning("Please enter a keyword.")
            else:
                with st.spinner("Fetching and analyzing posts..."):
                    posts = fetch_reddit_posts(keyword, num_posts)
                    st.session_state["social_posts"] = posts  # store in session

                    labels = []
                    for post in posts:
                        try:
                            res = detection_with_sentiment(post)
                            labels.append(res["status"])
                        except:
                            continue

                    if labels:
                        distribution = Counter(labels)
                        df = pd.DataFrame(
                            distribution.items(),
                            columns=["Mental Health Status", "Count"]
                        )
                        st.session_state["social_result"] = df
                    else:
                        st.session_state["social_result"] = None

        # Only show fetched posts if they exist
        if st.button("Show fetched posts" if language == "English" else "Tunjukkan Hantaran"):
            posts = st.session_state.get("social_posts", [])
            if posts:
                for i, post in enumerate(posts[:10]):  # show top 10
                    st.markdown(f"**Post {i+1}:** {post}")
            else:
                st.warning("No posts fetched yet. Click 'Analyze Social Media' first."
                        if language == "English"
                        else "Tiada hantaran dijumpai. Sila klik 'Analisis Media Sosial' dahulu.")

        # Display pie chart if analysis has been done
        if st.session_state.get("social_result") is not None:
            df = st.session_state["social_result"]

            # Create two tabs: Distribution & Posts
            sm_tabs = st.tabs(
                ["Distribution", "Posts"]
                if language == "English"
                else ["Taburan", "Hantaran"]
            )

            with sm_tabs[0]:
                fig = px.pie(
                    df,
                    names="Mental Health Status",
                    values="Count",
                    title=(
                        "Mental Health Status Distribution from Social Media Posts"
                        if language == "English"
                        else "Taburan Status Kesihatan Mental dari Hantaran Media Sosial"
                    ),
                )
                st.plotly_chart(fig, width='stretch')

            with sm_tabs[1]:
                posts = st.session_state.get("social_posts", [])
                if posts:
                    for i, post in enumerate(posts[:20]):  # show top 20 posts
                        st.markdown(f"**Post {i+1}:** {post}")
                else:
                    st.info(
                        "No posts fetched yet. Click 'Analyze Social Media' first."
                        if language == "English"
                        else "Tiada hantaran dijumpai. Sila klik 'Analisis Media Sosial' dahulu."
                    )
