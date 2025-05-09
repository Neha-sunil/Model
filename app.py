# app.py
import os
import streamlit as st
st.set_page_config(page_title="Cardio Disease Predictor", layout="wide")
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import speech_recognition as sr
from pydub import AudioSegment
import io
from audiorecorder import audiorecorder

# Configuration - Force CPU usage
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cpu")  # Force CPU only

# Load model and tokenizer with caching
@st.cache_resource
def load_model():
    model_name = "Tufan1/BioClinicalBERT-Cardio-Classifier-Fold-per1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1
    ).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Medical mappings
MEDICAL_MAPPINGS = {
    'gender': {1: "Female", 2: "Male"},
    'cholesterol': {1: "Normal", 2: "Elevated", 3: "Peak"},
    'glucose': {1: "Normal", 2: "High", 3: "Extreme"},
    'binary': {0: "No", 1: "Yes"}
}

# Streamlit UI
#st.set_page_config(page_title="Cardio Disease Predictor", layout="wide")
st.title("🫀 Cardiovascular Disease Prediction")

def safe_extract(pattern, text, group=1, default=None):
    """Regex helper function"""
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(group).lower() if match else default

def text_to_features(text):
    """Extract features from text input"""
    patterns = {
        'age': r'(\d+)\s*(?:years?|yrs?|year-old)',
        'gender': r'\b(male|female|man|woman)\b',
        'height': r'(?:height|ht)\D*(\d+)\s*cm',
        'weight': r'(?:weight|wt)\D*(\d+)\s*kg',
        'bp': r'(?:blood pressure|bp)\D*(\d+)\s*/\s*(\d+)',
        'cholesterol': r'(cholest(?:erol)?)\s*(?:is|level)?\s*(normal|elevated|peak)',
        'glucose': r'(glucose|sugar)\s*(?:is|level)?\s*(normal|high|extreme)',
        'smoke': r'\b(smokes?|smoking|smoke)\b',
        'alco': r'\b(alcohol|drinks?)\b',
        'active': r'\b(active|exercise|exercises)\b'
    }
    
    features = {
        'age': int(safe_extract(patterns['age'], text, 1, '0')) or None,
        'gender': 2 if safe_extract(patterns['gender'], text, 1, '').lower() in ['male', 'man'] else 1,
        'height': int(safe_extract(patterns['height'], text, 1, '0')) or None,
        'weight': int(safe_extract(patterns['weight'], text, 1, '0')) or None,
        'ap_hi': None,
        'ap_lo': None,
        'cholesterol': 1,
        'glucose': 1,
        'smoke': 1 if safe_extract(patterns['smoke'], text) else 0,
        'alco': 1 if safe_extract(patterns['alco'], text) else 0,
        'active': 1 if safe_extract(patterns['active'], text) else 0
    }

    # BP handling
    bp_match = re.search(patterns['bp'], text, re.IGNORECASE)
    if bp_match and len(bp_match.groups()) == 2:
        features['ap_hi'], features['ap_lo'] = map(int, bp_match.groups())

    # Cholesterol/glucose mapping
    chol_value = safe_extract(patterns['cholesterol'], text, 2)
    if chol_value:
        features['cholesterol'] = {'normal':1, 'elevated':2, 'peak':3}.get(chol_value, 1)
    
    gluc_value = safe_extract(patterns['glucose'], text, 2)
    if gluc_value:
        features['glucose'] = {'normal':1, 'high':2, 'extreme':3}.get(gluc_value, 1)

    return features

def process_audio(audio_file):
    """Handle audio uploads"""
    try:
        audio = AudioSegment.from_file(audio_file)
        audio_io = io.BytesIO()
        audio.export(audio_io, format="wav")
        audio_io.seek(0)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None

def predict(features):
    """Make prediction"""
    input_text = f"""Patient Record:
- Age: {features['age']} years
- Gender: {MEDICAL_MAPPINGS['gender'][features['gender']]}
- Height: {features['height']} cm
- Weight: {features['weight']} kg
- BP: {features['ap_hi']}/{features['ap_lo']} mmHg
- Cholesterol: {MEDICAL_MAPPINGS['cholesterol'][features['cholesterol']]}
- Glucose: {MEDICAL_MAPPINGS['glucose'][features['glucose']]}
- Smoke: {MEDICAL_MAPPINGS['binary'][features['smoke']]}
- Alco: {MEDICAL_MAPPINGS['binary'][features['alco']]}
- Active: {MEDICAL_MAPPINGS['binary'][features['active']]}"""

    inputs = tokenizer(
        input_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.sigmoid(logits).item()

    return "Cardiovascular Disease" if prob >= 0.5 else "No Cardiovascular Disease", round(prob*100, 2)

# Main UI
input_type = st.radio("Select input method:", ["Manual Input", "Text Description", "Audio Recording"])

if input_type == "Manual Input":
    with st.form("manual_input"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 10, 120)
            height = st.number_input("Height (cm)", 50, 250)
            ap_hi = st.number_input("Systolic BP", 80, 250)
            cholesterol = st.selectbox("Cholesterol", ["Normal", "Elevated", "Peak"])
            smoke = st.selectbox("Smoke", ["Yes", "No"])
            active = st.selectbox("Active", ["Yes", "No"])
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            weight = st.number_input("Weight (kg)", 30, 300)
            ap_lo = st.number_input("Diastolic BP", 40, 150)
            glucose = st.selectbox("Glucose", ["Normal", "High", "Extreme"])
            alco = st.selectbox("Alco", ["Yes", "No"])
        
        submitted = st.form_submit_button("Predict")
        if submitted:
            features = {
                'age': age,
                'gender': 1 if gender == "Female" else 2,
                'height': height,
                'weight': weight,
                'ap_hi': ap_hi,
                'ap_lo': ap_lo,
                'cholesterol': ["Normal", "Elevated", "Peak"].index(cholesterol) + 1,
                'glucose': ["Normal", "High", "Extreme"].index(glucose) + 1,
                'smoke': 1 if smoke == "Yes" else 0,
                'alco': 1 if alco == "Yes" else 0,
                'active': 1 if active == "Yes" else 0,

            }
            prediction, confidence = predict(features)
            st.success(f"**Prediction:** {prediction} (Confidence: {confidence}%)")

elif input_type == "Text Description":
    text_input = st.text_area("Enter patient description:", height=150)
    if st.button("Analyze Text"):
        features = text_to_features(text_input)
        if all(v is not None for v in features.values()):
            prediction, confidence = predict(features)
            st.success(f"**Prediction:** {prediction} (Confidence: {confidence}%)")
        else:
            st.warning("Could not extract all required parameters from text")

elif input_type == "Audio Recording":
    st.subheader("🎙️ Record or Upload Audio")

    # Upload audio file
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])

    # Record audio using mic (Streamlit extension)
    try:
        #from streamlit_audiorecorder import audiorecorder
        recorded_audio = audiorecorder("Start Recording", "Stop Recording")

        if recorded_audio is not None and len(recorded_audio) > 0:
            if st.button("Transcribe & Predict (Recorded Audio)"):
                try:
                    # If the recorder returns AudioSegment, convert to bytes
                    if isinstance(recorded_audio, AudioSegment):
                        audio_io = io.BytesIO()
                        recorded_audio.export(audio_io, format="wav")
                        audio_bytes = audio_io.getvalue()
                    else:
                        audio_bytes = recorded_audio

                    # Playback
                    st.audio(audio_bytes, format="audio/wav")

                    # Speech recognition
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)
                        st.markdown("**Transcribed Text:**")
                        st.info(text)

                        features = text_to_features(text)
                        missing = [k for k, v in features.items() if v is None]

                        if missing:
                            st.warning(f"Missing fields: {', '.join(missing)}. Using default values.")
                            features.setdefault("age", 50)
                            features.setdefault("gender", 2)
                            features.setdefault("height", 165)
                            features.setdefault("weight", 70)
                            features.setdefault("ap_hi", 120)
                            features.setdefault("ap_lo", 80)
                            features.setdefault("cholesterol", 1)
                            features.setdefault("glucose", 1)
                            features.setdefault("smoke", 0)
                            features.setdefault("alco", 1)
                            features.setdefault("active", 1)

                        prediction, confidence = predict(features)
                        st.success(f"**Prediction:** {prediction} (Confidence: {confidence}%)")

                except Exception as e:
                    st.error(f"Recording Processing Error: {e}")

    except Exception as e:
        st.info("🎙️ Audio recording not supported in this environment.")

    # Uploaded audio
    if audio_file is not None:
        if st.button("Transcribe & Predict (Uploaded Audio)"):
            try:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format=audio_file.type)

                recognizer = sr.Recognizer()
                with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    st.markdown("**Transcribed Text:**")
                    st.info(text)

                    features = text_to_features(text)
                    missing = [k for k, v in features.items() if v is None]

                    if missing:
                        st.warning(f"Missing fields: {', '.join(missing)}. Using default values.")
                        features.setdefault("age", 50)
                        features.setdefault("gender", 2)
                        features.setdefault("height", 165)
                        features.setdefault("weight", 70)
                        features.setdefault("ap_hi", 120)
                        features.setdefault("ap_lo", 80)
                        features.setdefault("cholesterol", 1)
                        features.setdefault("glucose", 1)
                        features.setdefault("smoke", 0)
                        features.setdefault("alco", 1)
                        features.setdefault("active", 1)

                    prediction, confidence = predict(features)
                    st.success(f"**Prediction:** {prediction} (Confidence: {confidence}%)")

            except Exception as e:
                st.error(f"Uploaded Audio Error: {e}")
