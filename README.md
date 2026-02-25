# Sona AI 🎙️✨

Sona AI is an interview-focused audio transcription and summarization platform. It leverages state-of-the-art AI models to provide fast, accurate transcription with speaker diarization, followed by summarization of the interview.

---

## ✨ Features

- **Transcription**: Powered by [WhisperX](https://github.com/m-bain/whisperX) for faster-whisper inference and precise word-level alignment.
- **Speaker Diarization**: Automatically detects, identifies, and labels different speakers in your audio files.
- **Summarization**: Uses **Llama** to generate concise and meaningful summaries of transcribed conversations.
- **Multilingual Support**: Supports English, Indonesian, and automatic language detection.
- **Frontend Support**: Uses [Next.js](https://nextjs.org/) for a modern, responsive web interface.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** & **npm**
- **Hugging Face Token**: Required for speaker diarization models (pyannote). Get it [here](https://huggingface.co/settings/tokens).

### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wikananda/sona-ai.git
   cd sona-ai
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Hugging Face token:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Run the API**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **ASR Engine**: [WhisperX](https://github.com/m-bain/whisperX) (faster-whisper + wav2vec2 alignment)
- **Diarization**: [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- **Summarization**: [Hugging Face Transformers](https://huggingface.co/meta-llama) (Llama)
- **Deep Learning**: PyTorch

### Frontend
- **Framework**: [Next.js 15](https://nextjs.org/) (App Router)
- **Language**: TypeScript
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/)

---

## 📂 Project Structure

```text
sona-ai/
├── api/                # FastAPI application & endpoints
├── transcription/      # WhisperX engine implementation
├── summarization/      # Llama training and inference logic
├── configs/            # Model and application configurations (.yaml)
├── frontend/           # Next.js web application
├── utils/              # Helper functions and utilities
├── data/               # Local data storage (raw audio, etc.)
└── outputs/            # Generated transcripts and summaries
```

---

## 📖 Usage

### Transcription API
Send a `POST` request to `/transcribe` with an audio file:
- **Endpoint**: `http://localhost:8000/transcribe`
- **Parameters**:
    - `file`: Audio file (mp3, wav, etc.)
    - `language`: (Optional) Language code (e.g., `en`, `id`)
    - `min_speakers`: (Optional) Minimum number of speakers
    - `max_speakers`: (Optional) Maximum number of speakers

### Summarization API
Send a `POST` request to `/summarize` with a transcript:
- **Endpoint**: `http://localhost:8000/summarize`
- **Parameters**:
    - `text`: The transcript to summarize
    - `prompt`: (Optional) Custom instruction prompt
    - `max_length`: (Optional) Maximum token length for the input

### Web Interface
1. Upload an audio file through the dashboard.
2. Select the language or use auto-detection.
3. Configure speaker settings if necessary.
4. Click **Upload & Transcribe** to generate transcription.
5. Click **Summarize** to generate summary once the transcription is generated.

---

## 📝 TODO
- [x] Implement basic API for transcription and summarization
- [x] Implement basic UI for transcription and summarization
- [ ] UI Overhaul