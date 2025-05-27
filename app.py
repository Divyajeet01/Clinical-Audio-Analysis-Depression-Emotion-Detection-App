import streamlit as st
import numpy as np
import pandas as pd
import io
import librosa
import torch
import torchaudio
import joblib
import opensmile
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import matplotlib.pyplot as plt

# === Streamlit Page Config ===
st.set_page_config(page_title="Audio Emotion & Depression Analysis", layout="wide")

# === Load Models ===
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    emotion_model = AutoModelForAudioClassification.from_pretrained(
        r"C:\Users\divya\OneDrive\Desktop\projects\Audio Depression\final\emotion_model"  # Ensure this path is valid
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        r"C:\Users\divya\OneDrive\Desktop\projects\Audio Depression\final\emotion_feature_extractor"  # Ensure this path is valid
    )
    return rf_model, scaler, emotion_model, feature_extractor

model, scaler, emotion_model, feature_extractor = load_models()
id2label = emotion_model.config.id2label

# === OpenSMILE Setup ===
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)

# === Feature Extraction for Depression ===
@st.cache_data(show_spinner=False)
def extract_depression_features(audio_bytes):
    with io.BytesIO(audio_bytes) as f:
        y, sr = librosa.load(f, sr=None)
        features = smile.process_signal(y, sr)
        df = pd.DataFrame(features)

        energy = librosa.feature.rms(y=y)[0]
        pause_threshold = np.percentile(energy, 5)
        pause_duration = np.sum(energy < pause_threshold) / sr
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        speech_rate = np.mean(zcr) * sr
        entropy = -np.sum(energy * np.log2(energy + 1e-10))

        df['Pause_Duration'] = pause_duration
        df['Speech_Rate'] = speech_rate
        df['Energy_Entropy'] = entropy

        stats = []
        for col in df.columns:
            series = df[col]
            stats.extend([series.mean(), series.std(), series.min(), series.max(), series.median()])
        return np.array(stats)

# === Chunk Audio into 1-minute segments ===
def chunk_audio(audio_bytes, chunk_duration=60):
    waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
    chunk_size = chunk_duration * sr
    buffers = []
    for i in range(0, waveform.shape[1], chunk_size):
        chunk = waveform[:, i:i + chunk_size]
        if chunk.shape[1] >= sr:
            buf = io.BytesIO()
            torchaudio.save(buf, chunk, sr, format="wav")
            buf.seek(0)
            buffers.append(buf)
    return buffers

# === Predict Emotions ===
def predict_emotions(chunks):
    results = []
    confidences = []
    overall_logits = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_model.to(device)

    for idx, buf in enumerate(chunks):
        y, sr = librosa.load(buf, sr=feature_extractor.sampling_rate)
        max_len = int(sr * 30.0)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)))
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = emotion_model(**inputs)
            logits = output.logits[0].cpu().numpy()

            if overall_logits is None:
                overall_logits = logits.copy()
            else:
                overall_logits += logits

            pred = np.argmax(logits)
            prob = torch.nn.functional.softmax(output.logits, dim=-1)[0][pred].cpu().item()
            label = id2label[pred]
            results.append((f"{idx+1}", label, prob * 100))
            confidences.append((f"{idx+1}", label, f"{prob*100:.2f}%"))

    avg_probs = overall_logits / len(chunks) if chunks else overall_logits
    return results, confidences, avg_probs

# === Plot Waveform ===
def plot_waveform(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    times = np.linspace(0, duration, num=len(y))
    fig, ax = plt.subplots(figsize=(10, 3))  
    ax.plot(times, y, label="Waveform")  
    ax.set_title("Audio Waveform")  
    ax.set_xlabel("Time (s)")  
    ax.set_ylabel("Amplitude")  
    ax.grid(True)  
    st.pyplot(fig)

# === Plot Emotion Pie Chart ===
def plot_emotion_pie_chart(emotion_results):
    emotion_counts = {}
    for _, emotion, _ in emotion_results:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1

    labels = list(emotion_counts.keys())  
    sizes = list(emotion_counts.values())  
    fig, ax = plt.subplots(figsize=(6, 6))  
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])  
    ax.axis('equal')  
    ax.set_title('Emotion Distribution (Pie Chart)')  
    st.pyplot(fig)

# === Main Streamlit App ===
st.title("🎧 Audio-based Depression & Emotion Analysis")
uploaded = st.file_uploader("Upload a .wav audio file", type=["wav"])

# Reset state on new upload
if uploaded:
    if 'last_file_name' not in st.session_state or st.session_state['last_file_name'] != uploaded.name:
        st.session_state.clear()
        st.session_state['last_file_name'] = uploaded.name

if uploaded:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")

    # Display Waveform  
    plot_waveform(audio_bytes)  

    # === Depression Prediction ===
    if 'depression' not in st.session_state:  
        with st.spinner("Analyzing depression..."):  
            feats = extract_depression_features(audio_bytes)  
            scaled = scaler.transform([feats])  
            probs = model.predict_proba(scaled)[0]  
            pred = np.argmax(probs)  
            label = "Depressed" if pred == 1 else "Not Depressed"  
            conf = probs[pred] * 100  
            st.session_state['depression'] = (label, conf)  

    label, conf = st.session_state['depression']  
    st.success(f"🧠 Depression Prediction: *{label}*")  
    st.info(f"Confidence: *{conf:.2f}%*")  

    # === Emotion Prediction Button ===
    if st.button("Predict Emotions per Minute"):  
        with st.spinner("Chunking and predicting emotions..."):  
            chunks = chunk_audio(audio_bytes)  
            res, confs, dist = predict_emotions(chunks)  
            st.session_state['emotion_results'] = res  
            st.session_state['emotion_confidences'] = confs  
            st.session_state['overall_dist'] = dist  
            st.session_state['show_vis'] = False  

    # === Display Emotion Results ===
    if 'emotion_results' in st.session_state:  
        st.markdown("---")
        st.subheader("Minute-wise Emotion Prediction")  
        df_conf = pd.DataFrame(st.session_state['emotion_confidences'],  
                               columns=["Minute", "Emotion", "Confidence"])  
        st.dataframe(df_conf)  

        st.subheader("Emotion Distribution")  
        plot_emotion_pie_chart(st.session_state['emotion_results'])  

       
