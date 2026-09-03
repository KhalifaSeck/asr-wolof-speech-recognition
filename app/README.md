# `app/` — Streamlit demo

Interactive web UI that wraps the fine-tuned Wolof ASR model behind a polished glassmorphism interface. Upload an audio file, get its Wolof transcription in seconds.

The app consumes the model published on Hugging Face
([`Bakis/ASR-Model-Wav2vec2`](https://huggingface.co/Bakis/ASR-Model-Wav2vec2))
— **no training required to run it**.

---

## Run locally

### With Python

```bash
cd app
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.
First launch downloads the ~1.3 GB model from Hugging Face (cached in `/tmp/huggingface`).

### With Docker

**GPU:**
```bash
cd app
docker build -t asr-wolof-app .
docker run --rm -p 8501:8501 --gpus all asr-wolof-app
```

**CPU-only:** open `streamlit_app.py` and change `device=0` to `device=-1` in `get_asr_pipeline()`.

---

## Deploy

The container listens on port `8501` and exposes `/_stcore/health`, so it drops into any modern platform:

| Platform | Notes |
|---|---|
| **Hugging Face Spaces** | *Docker* template, push the `app/` folder as a Space. Free CPU tier. |
| **Streamlit Community Cloud** | Point at `app/streamlit_app.py`. Free, CPU only. |
| **Render / Railway / Fly.io** | Auto-detects the Dockerfile. |
| **Kubernetes** | Build and push, then a Deployment + Service on port 8501. |

---

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Main application (UI + inference glue) |
| `requirements.txt` | Runtime dependencies |
| `Dockerfile` | Production-ready container |
| `.dockerignore` | Slim build context |

---

For the training pipeline behind the model, see the parent project
[`asr-wolof-speech-recognition`](../).
