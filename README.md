# Pipecat RAG Voice Bot

A **Retrieval-Augmented Generation (RAG)** voice AI bot built with [Pipecat](https://github.com/pipecat-ai/pipecat). Users speak naturally into their browser, and the bot answers questions grounded in the contents of a PDF or DOCX document — all in real time.

---

## Table of Contents

- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Environment Configuration](#environment-configuration)
- [How to Run](#how-to-run)
- [Deploy to Production](#deploy-to-production)

---

## Introduction

This project is a voice-powered question-answering assistant. Instead of relying on the LLM's pre-trained knowledge, it retrieves relevant passages from a user-supplied document (PDF or DOCX) and uses them as context when generating answers. This ensures responses are accurate, grounded, and traceable back to source material.

Key capabilities:

- **Voice-in, voice-out** — speak your question and hear the answer.
- **Document-grounded answers** — the bot only responds based on your uploaded document.
- **Interview question generation** — ask for interview questions derived from a resume or any document.
- **Local LLM support** — runs inference through [LM Studio](https://lmstudio.ai/) so no data leaves your machine.

---

## How It Works

```
┌──────────┐    audio    ┌────────────┐   text    ┌───────────┐
│  Browser  │ ────────► │ Deepgram   │ ───────► │  RAG      │
│  (WebRTC) │           │ STT        │          │  Retriever│
└──────────┘           └────────────┘          └─────┬─────┘
      ▲                                              │ context + query
      │ audio                                        ▼
┌──────────┐           ┌────────────┐          ┌───────────┐
│ Cartesia │ ◄──────── │  LLM       │ ◄─────── │  Prompt   │
│ TTS      │   text    │ (LM Studio)│          │  Builder  │
└──────────┘           └────────────┘          └───────────┘
```

1. **Speech-to-Text** — Deepgram transcribes the user's spoken question in real time.
2. **Retrieval** — The query is embedded using `all-MiniLM-L6-v2` and compared against pre-computed document chunk embeddings via cosine similarity. The top-k most relevant chunks are selected.
3. **Augmented Prompt** — Retrieved chunks are injected into the system prompt alongside the user's question.
4. **LLM Inference** — The augmented prompt is sent to a local LLM served by LM Studio (OpenAI-compatible API).
5. **Text-to-Speech** — Cartesia converts the LLM's response into natural-sounding audio streamed back to the browser.

---

## Architecture

| Component        | Technology                                 | Purpose                                                 |
| ---------------- | ------------------------------------------ | ------------------------------------------------------- |
| Transport        | Pipecat WebRTC                             | Real-time browser ↔ server audio streaming              |
| STT              | Deepgram                                   | Converts speech to text                                 |
| Embeddings       | Sentence-Transformers (`all-MiniLM-L6-v2`) | Encodes document chunks & queries for similarity search |
| Document Parsing | PyMuPDF / python-docx                      | Extracts text from PDF and DOCX files                   |
| LLM              | LM Studio (OpenAI-compatible)              | Generates answers from augmented context                |
| TTS              | Cartesia                                   | Converts text responses to speech                       |
| VAD              | Silero                                     | Voice Activity Detection for turn-taking                |

---

## Requirements

### System

- **Python** 3.10 or later
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** package manager
- **[LM Studio](https://lmstudio.ai/)** installed and running with a loaded model
- A PDF or DOCX document to use as the knowledge base

### API Keys

| Service  | Purpose        | Sign-up Link                                                |
| -------- | -------------- | ----------------------------------------------------------- |
| Deepgram | Speech-to-Text | [console.deepgram.com](https://console.deepgram.com/signup) |
| Cartesia | Text-to-Speech | [play.cartesia.ai](https://play.cartesia.ai/sign-up)        |

> **Note:** No OpenAI API key is required. LLM inference runs locally through LM Studio.

### Python Dependencies

Core dependencies are declared in `pyproject.toml`:

```
pipecat-ai[webrtc,daily,silero,deepgram,openai,cartesia,runner]
pipecat-ai-cli
```

Additional libraries used at runtime:

```
PyMuPDF          # PDF text extraction
python-docx      # DOCX text extraction
sentence-transformers  # Embedding model
numpy            # Similarity computation
python-dotenv    # .env file loading
loguru           # Logging
```

---

## Environment Configuration

1. **Copy the example file** to create your local `.env`:

   ```bash
   cp env.example .env
   ```

2. **Edit `.env`** and fill in each variable:

   ```ini
   # ── Speech-to-Text ─────────────────────────────────────
   DEEPGRAM_API_KEY=your_deepgram_api_key_here

   # ── LLM (not used — inference is local via LM Studio) ──
   OPENAI_API_KEY=your_openai_api_key_here

   # ── Text-to-Speech ─────────────────────────────────────
   CARTESIA_API_KEY=your_cartesia_api_key_here

   # ── LM Studio local server ─────────────────────────────
   LM_STUDIO_BASE_URL=http://localhost:1234/v1
   MODEL_NAME=your_model_name_here
   ```

   | Variable             | Description                                                                          |
   | -------------------- | ------------------------------------------------------------------------------------ |
   | `DEEPGRAM_API_KEY`   | Your Deepgram API key for real-time speech recognition.                              |
   | `OPENAI_API_KEY`     | Placeholder — can be any value (e.g., `not-needed`) since the LLM is local.          |
   | `CARTESIA_API_KEY`   | Your Cartesia API key for text-to-speech synthesis.                                  |
   | `LM_STUDIO_BASE_URL` | URL where LM Studio's local server is running (default: `http://localhost:1234/v1`). |
   | `MODEL_NAME`         | The exact model identifier loaded in LM Studio (e.g., `mistral-7b-instruct`).        |

3. **Place your document** in the project root. By default the bot loads `Resume_BD.pdf`. To change this, edit the `doc_file` variable in `bot.py`:

   ```python
   doc_file = "your_document.pdf"  # or "your_document.docx"
   ```

---

## How to Run

### 1. Start LM Studio

- Open LM Studio and load your preferred model.
- Start the local server (default: `http://localhost:1234/v1`).

### 2. Install Dependencies

```bash
uv sync
```

### 3. Launch the Bot

```bash
uv run bot.py
```

### 4. Connect from Browser

Open **http://localhost:7860** and click **Connect** to begin the voice conversation.

> **First run:** Initial startup may take ~20–30 seconds while Pipecat downloads the Silero VAD model and `all-MiniLM-L6-v2` embeddings.

---

## Deploy to Production

Transform your local bot into a production-ready service. Pipecat Cloud handles scaling, monitoring, and global deployment.

### Prerequisites

1. [Sign up for Pipecat Cloud](https://pipecat.daily.co/sign-up).

2. Set up Docker for building your bot image:
   - **Install [Docker](https://www.docker.com/)** on your system
   - **Create a [Docker Hub](https://hub.docker.com/) account**
   - **Login to Docker Hub:**

     ```bash
     docker login
     ```

3. Install the Pipecat CLI

   ```bash
   uv tool install pipecat-ai-cli
   ```

   > Tip: You can run the `pipecat` CLI using the `pc` alias.

### Configure your deployment

The `pcc-deploy.toml` file tells Pipecat Cloud how to run your bot. **Update the image field** with your Docker Hub username by editing `pcc-deploy.toml`.

```ini
agent_name = "quickstart"
image = "YOUR_DOCKERHUB_USERNAME/quickstart:0.1"  # 👈 Update this line
secret_set = "quickstart-secrets"

[scaling]
	min_agents = 1
```

**Understanding the TOML file settings:**

- `agent_name`: Your bot's name in Pipecat Cloud
- `image`: The Docker image to deploy (format: `username/image:version`)
- `secret_set`: Where your API keys are stored securely
- `min_agents`: Number of bot instances to keep ready (1 = instant start)

> 💡 Tip: [Set up `image_credentials`](https://docs.pipecat.ai/deployment/pipecat-cloud/fundamentals/secrets#image-pull-secrets) in your TOML file for authenticated image pulls

### Log in to Pipecat Cloud

To start using the CLI, authenticate to Pipecat Cloud:

```bash
pipecat cloud auth login
```

You'll be presented with a link that you can click to authenticate your client.

### Configure secrets

Upload your API keys to Pipecat Cloud's secure storage:

```bash
pipecat cloud secrets set quickstart-secrets --file .env
```

This creates a secret set called `quickstart-secrets` (matching your TOML file) and uploads all your API keys from `.env`.

### Build and deploy

Build your Docker image and push to Docker Hub:

```bash
pipecat cloud docker build-push
```

Deploy to Pipecat Cloud:

```bash
pipecat cloud deploy
```

### Connect to your agent

1. Open your [Pipecat Cloud dashboard](https://pipecat.daily.co/)
2. Select your `quickstart` agent → **Sandbox**
3. Allow microphone access and click **Connect**

---

## What's Next?

**🔧 Customize your bot**: Modify `bot.py` to change personality, add functions, or integrate with your data  
**📚 Learn more**: Check out [Pipecat's docs](https://docs.pipecat.ai/) for advanced features  
**💬 Get help**: Join [Pipecat's Discord](https://discord.gg/pipecat) to connect with the community

### Troubleshooting

- **Browser permissions**: Allow microphone access when prompted
- **Connection issues**: Try a different browser or check VPN/firewall settings
- **Audio issues**: Verify microphone and speakers are working and not muted
