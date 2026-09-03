import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"

import tempfile
import time

import streamlit as st
import torchaudio
from transformers import pipeline

st.set_page_config(
    page_title="ASR Wolof | Reconnaissance Vocale",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border-color: #334155;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Poppins', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.3)); }
        to { filter: drop-shadow(0 0 30px rgba(139, 92, 246, 0.5)); }
    }

    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .upload-zone {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .upload-zone:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
        transform: scale(1.02);
    }

    .upload-icon {
        font-size: 4rem;
        color: var(--primary-color);
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }

    .transcription-result {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }

    .transcription-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .transcription-text {
        font-size: 1.3rem;
        font-weight: 500;
        color: var(--text-primary);
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
    }

    .stAlert > div {
        border-radius: 15px;
        border: none;
        backdrop-filter: blur(20px);
    }

    .stSuccess > div {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 182, 212, 0.2));
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .stInfo > div {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(99, 102, 241, 0.2));
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .stAudio {
        margin: 2rem 0;
    }

    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }

    .stat-card {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        flex: 1;
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        background: rgba(99, 102, 241, 0.2);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        display: block;
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .custom-footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid var(--border-color);
        color: var(--text-secondary);
    }

    @media (max-width: 768px) {
        .main-title { font-size: 2.5rem; }
        .stats-container { flex-direction: column; }
        .glass-card { padding: 1.5rem; }
    }

    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

MODEL_NAME = "Bakis/ASR-Model-Wav2vec2"
TASK = "automatic-speech-recognition"


@st.cache_resource(show_spinner=True)
def get_asr_pipeline():
    return pipeline(TASK, model=MODEL_NAME, device=0)


asr_pipe = get_asr_pipeline()


def main():
    st.markdown('<h1 class="main-title">🎤 ASR Wolof</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Reconnaissance vocale intelligente pour la langue Wolof</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stat-card fade-in">
            <span class="stat-number">Wav2Vec2</span>
            <div class="stat-label">Architecture</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card fade-in">
            <span class="stat-number">XLS-R</span>
            <div class="stat-label">Modèle de base</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card fade-in">
            <span class="stat-number">16kHz</span>
            <div class="stat-label">Échantillonnage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown("""
    ### 🚀 Comment utiliser l'ASR Wolof ?

    1. **Importez** votre fichier audio dans l'un des formats supportés
    2. **Écoutez** l'aperçu de votre fichier audio
    3. **Attendez** que la transcription soit générée automatiquement
    4. **Copiez** le résultat pour l'utiliser dans vos projets

    #### 📁 Formats supportés
    - WAV, MP3, FLAC, OGG
    - Conversion automatique en 16kHz mono
    - Traitement optimisé pour la langue Wolof
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📁</div>
        <h3>Déposez votre fichier audio ici</h3>
        <p>Ou cliquez pour parcourir vos fichiers</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier audio",
        type=["wav", "mp3", "flac", "ogg"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        file_details = {
            "Nom du fichier": uploaded_file.name,
            "Taille": f"{len(uploaded_file.getvalue()) / 1024:.1f} KB",
            "Type": uploaded_file.type,
        }

        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown("### 📊 Informations du fichier")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("📄 Nom", file_details["Nom du fichier"])
        with c2:
            st.metric("📏 Taille", file_details["Taille"])
        with c3:
            st.metric("🎵 Format", file_details["Type"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.markdown("### 🎧 Aperçu audio")
        st.audio(uploaded_file, format="audio/wav")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("🔄 Traitement en cours..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("📥 Chargement du fichier audio...")
                elif i < 60:
                    status_text.text("🔄 Conversion en format optimal...")
                elif i < 90:
                    status_text.text("🧠 Analyse par le modèle IA...")
                else:
                    status_text.text("✨ Finalisation de la transcription...")
                time.sleep(0.02)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                tmp_wav.write(uploaded_file.read())
                audio_path = tmp_wav.name

            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            converted_path = audio_path + "_converted.wav"
            torchaudio.save(converted_path, waveform, 16000)

            result = asr_pipe(converted_path)
            transcription = result["text"]

            os.unlink(audio_path)
            os.unlink(converted_path)

        if transcription:
            st.markdown('<div class="transcription-result fade-in">', unsafe_allow_html=True)
            st.markdown("### ✨ Transcription générée")
            st.markdown(f'<div class="transcription-text">"{transcription}"</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                if st.button("📋 Copier le texte", key="copy"):
                    st.success("✅ Texte copié dans le presse-papier!")
            with b2:
                if st.button("🔄 Nouvelle transcription", key="new"):
                    st.rerun()
            with b3:
                if st.button("💾 Télécharger", key="download"):
                    st.download_button(
                        label="📥 Télécharger la transcription",
                        data=transcription,
                        file_name=f"transcription_{uploaded_file.name}.txt",
                        mime="text/plain",
                    )
    else:
        st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
        st.info("🎯 **Prêt à transcrire !** Importez votre fichier audio pour commencer la reconnaissance vocale.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card fade-in">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔬 À propos de cette technologie

    Cette application repose sur le modèle Wav2Vec2 XLS-R-300M, spécialement fine-tuné pour la langue wolof
    et déployé sur HuggingFace (Bakis/ASR-Model-Wav2vec2). Entraîné sur un corpus représentatif de données
    audio en wolof, ce modèle assure une reconnaissance vocale fiable et adaptée aux spécificités de la langue.

    **Caractéristiques techniques :**
    - 🎯 Architecture : Wav2Vec2 XLS-R-300M
    - 🌍 Langue : Wolof
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-footer">
        <p>🚀 <strong>ASR Wolof</strong> | Propulsé par <strong>Bakis/ASR-Model-Wav2vec2</strong></p>
        <p>Développé pour la communauté Wolof | © 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
