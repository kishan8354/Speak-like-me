# 🗣️ Multilingual Speech-to-Speech Translation with Speaker Identity and Emotion Preservation

A **Streamlit-based AI system** for multilingual speech translation that preserves **speaker identity and emotional expression** using **Coqui XTTS v2**.

This system enables users to upload or record a reference voice and generate natural-sounding speech in multiple languages while maintaining:

- 🎙️ Voice identity  
- 🎭 Emotional tone  
- 🌍 Cross-lingual consistency  

Designed as part of an advanced research project in AI-driven speech synthesis and expressive voice conversion.

---

## 🚀 Key Features

### 🎙️ Speaker Identity Preservation
- High speaker similarity using cosine embedding matching
- Adjustable similarity & stability controls
- Voice cloning with reference-based synthesis

### 🌍 Multilingual Speech Translation
- Supports 16+ languages  
- Cross-lingual voice transfer  
- Natural pronunciation in target language  

### 🎭 Emotion-Aware Synthesis
- Emotion controls: happy, sad, angry, surprised, fearful
- Prosody transfer for expressive speech
- Emotion confidence tracking

### 🎛 Advanced Controls
- Voice stability tuning
- Similarity boost option
- Loudness normalization
- Speed and expressive mode selection

### 📊 Evaluation Metrics
- Speaker Similarity (Cosine)
- Signal-to-Noise Ratio (SNR)
- Spectral Clarity
- Emotion Transfer Accuracy

### 🗂 Voice Management System
- Upload, record, activate, delete voices
- Batch delete option
- Preview reference voices
- Generation history tracking

### 💾 Export Options
- WAV output
- MP3 export (requires FFmpeg)

---

## 🛠 Tech Stack

### Frontend
- **Streamlit** – Interactive web UI  
- streamlit-option-menu – Sidebar navigation  
- Custom CSS (Dark Theme)  
- HTML5 MediaRecorder API  

### Backend
- **Coqui TTS (XTTS v2)** – Core voice cloning engine  
- librosa & soundfile – Audio processing  
- pydub – Format conversion  
- NumPy – Numerical processing  

### System Dependencies
- FFmpeg – Audio encoding/decoding  

---

## 📂 Project Structure

```
multilingual_speech_translation/
│
├── outputs/                  # Generated speech outputs
├── voices/                   # Reference voices
│
├── app.py                    # Main Streamlit application
├── run_app.py                # Cross-platform launcher
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / Mac
.venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Install FFmpeg

Linux:
```bash
sudo apt-get install ffmpeg
```

Mac:
```bash
brew install ffmpeg
```

Windows:
```bash
choco install ffmpeg
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

App runs at:

```
http://localhost:8502
```

---

## 📊 Research Contribution

This project explores:

- Cross-lingual voice identity preservation
- Emotion-aware speech synthesis
- Prosody transfer in multilingual settings
- Post-processing for perceptual enhancement

---

## 👨‍💻 Author

**Kishan Kushavaha**  
M.Tech – Computer Science  
Specialization: AI & Speech Systems  

GitHub: https://github.com/your-username  
LinkedIn: https://linkedin.com/in/your-profile  

---

## 📜 License

This project is licensed under the MIT License.
