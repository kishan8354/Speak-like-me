# 🗣️ AI Voice Cloner (XTTS v2)

A **Streamlit web app** for cloning voices using [Coqui TTS](https://github.com/coqui-ai/TTS).  
Upload or record reference voices, fine-tune style & settings, and generate natural-sounding speech in 16+ languages — all through a clean, dark-themed interface.

---

## 🚀 Features

- 📁 **Voice gallery management**
  - Upload, record, save, activate, and delete reference voices
  - Batch delete multiple voices at once
  - Preview and switch between saved voices
- 🌍 **Multi-language support**
  - 16+ languages including English, Spanish, French, German, Chinese, Japanese, and more
- 🎭 **Voice style controls**
  - Choose between neutral, fast, and expressive styles
  - Optional speed and emotion overrides (happy, sad, angry, surprised, fearful)
- ⚙️ **Advanced settings**
  - Control voice stability and similarity for better cloning accuracy
  - Toggle loudness normalization for consistent output
- 📊 **Generation history**
  - Automatic metadata saving (text, style, reference voice, timestamp)
  - Recent generations gallery with preview, download, and re-synthesize option
- 💾 **Multiple export formats**
  - Download results as WAV or MP3 (requires FFmpeg)
- 🎨 **Modern dark UI**
  - Custom-styled Streamlit interface with smooth UX and mobile-friendly layout

---

## 🛠 Tech Stack

### **Frontend / UI**

- [Streamlit](https://streamlit.io/): Web interface for interaction and real-time updates
- **streamlit-option-menu**: Sidebar navigation with a clean UI
- Custom CSS styling for dark mode & better UX
- HTML5 `MediaRecorder` for in-browser recording

### **Backend / Logic**

- [Coqui TTS (XTTS v2)](https://github.com/coqui-ai/TTS): Voice cloning & text-to-speech model
- **pydub**: Audio file conversion (WebM/MP3 ↔ WAV)
- **librosa & soundfile**: Audio signal processing
- **NumPy**: Array and numerical operations

### **Utilities**

- **ffmpeg (system dependency)**: Required for audio encoding/decoding
- **streamlit.components.v1** – custom recorder integration

---

## 📂 Project Structure

```text
voice_clone_app/
│
├── outputs/                  # Generated outputs (auto-created)
│   └── xtts_*.wav            # Generated files
│
├── voices/                   # Reference voices (auto-created)
│   ├── ref_*.wav             # Uploaded/recorded samples
│
├── app.py                    # Main Streamlit application
├── run_app.py                # Launcher (cross-platform, auto-opens browser)
├── requirements.txt          # Python dependencies
├── LICENSE                   # Open-source license
├── .gitignore                # Ignored files/folders
└── README.md                 # Project documentation
```

---

## ⚙️ Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/EbrahimAR/AI-Voice-Cloner-XTTS-v2.git
cd AI-Voice-Cloner-XTTS-v2
```

### **2. Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate    # On Linux/Mac
.venv\Scripts\activate       # On Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Install system dependencies**

- **ffmpeg** is required for MP3 conversion:
  - Windows: `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

---

## ▶️ Run the app

```bash
streamlit run app.py
```

Or use the helper script:

```bash
python run_app.py
```

App will open at: http://localhost:8502

---

## 👨‍💻 Author

Ebrahim Abdul Raoof

[LinkedIn](https://www.linkedin.com/in/ebrahim-ar/) | [GitHub](https://github.com/EbrahimAR)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](https://github.com/EbrahimAR/AI-Voice-Cloner-XTTS-v2/blob/main/LICENSE) for details.
