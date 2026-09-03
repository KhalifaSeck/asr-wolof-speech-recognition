# Setup — asr-wolof-speech-recognition

## 1. Create the repository on GitHub

Go to https://github.com/new

- **Repository name** : `asr-wolof-speech-recognition`
- **Public** ✅
- **Do NOT** initialize with README, .gitignore or LICENSE (we already have them)
- Click **Create repository**

## 2. Clone it locally

```bash
cd "C:\Users\lifas\OneDrive\Documents\GitHub"
git clone https://github.com/KhalifaSeck/asr-wolof-speech-recognition.git
cd asr-wolof-speech-recognition
```

## 3. Copy the project content

```bash
Copy-Item -Path "C:\Users\lifas\AppData\Local\Temp\claude\C--Users-lifas-OneDrive-Documents-Data-Engineering-realtime-gaming-platform\bb5538e7-71a6-49ed-b6d6-19d5217161d2\scratchpad\asr-wolof\*" -Destination . -Recurse -Force
```

Verify with:

```bash
ls -Recurse -Name
```

You should see: `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, `notebooks/`, `src/`, `scripts/`, `assets/`, `.github/`, `data/`, `models/`, `reports/`.

## 4. Commit and push

```bash
git add .
git commit -m "feat: initial Wolof ASR pipeline (Wav2Vec2 XLS-R + IndabaX dataset)"
git branch -M main
git push -u origin main
```

## 5. Verify

Open **https://github.com/KhalifaSeck/asr-wolof-speech-recognition** — the README with banner and architecture diagram should render.

## 6. Update your GitHub profile

Since this new repo is now public, it will appear in your featured projects on `github.com/KhalifaSeck`. You can pin it via **Customize your pins** on your profile.

The main portfolio README already links to `https://github.com/KhalifaSeck/asr-wolof-speech-recognition` — no update needed there once this repo exists.

## Optional — reproduce the training

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

python -m scripts.train
```

A GPU with at least 12 GB of VRAM is recommended.
