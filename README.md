# AtaraxAI: Private, Offline Mental Health Companion

> **A privacy-first offline AI companion designed to provide CBT-based emotional support and real-time crisis detection without internet access — powered by Gemma 3n and ONNX.**


<img width="1108" height="731" alt="thumbnail" src="https://github.com/user-attachments/assets/c970a701-c553-4957-a686-c47b7737204c" />

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Key Features](#key-features)
- [How Gemma 3n Is Used](#how-gemma-3n-is-used)
- [Dataset & Fine-tuning](#dataset--fine-tuning)
- [Technical Stack](#technical-stack)
- [Target Impact](#target-impact)
- [Demo Video Link](#demo--screenshots)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Ethical Disclaimer](#ethical-disclaimer)
- [License](#license)
- [Contact](#contact)

---

## Overview

AtaraxAI is an open-source, privacy-first AI mental health companion that provides Cognitive Behavioral Therapy (CBT) support and real-time crisis detection — all entirely **offline**. Powered by the Gemma 3n LLM and ONNX fine-tuned models, AtaraxAI empowers users to journal, reflect, and build resilience, ensuring sensitive data never leaves their device.

---

## Problem Statement

Millions face barriers to quality mental health support: stigma, lack of access, privacy concerns, and the high cost of care. Online AI tools can help, but they often require sharing personal data over the internet, risking confidentiality and trust. There is a critical need for an **offline, privacy-focused mental health AI assistant** that can deliver actionable CBT guidance and proactively flag moments of crisis, especially in low-resource or high-stigma environments.

---

## Solution Overview

**AtaraxAI** delivers:

- **Completely Offline Operation:** All data and model inference happen locally; no internet or cloud needed.
- **CBT Guidance & Reflection:** Personalized, context-aware CBT responses using Gemma 3n, tailored to user emotions, tone, and coping preferences.
- **Crisis Detection:** Real-time detection of crisis-level entries using a fine-tuned ONNX model, with automatic escalation suggestions and resource prompts.
- **Privacy-First Design:** Journals and session data never leave the user’s device; users control, clear, and export their own data at any time.
- **Rich Analytics & Exports:** Visualize emotional trends, crisis risk, and session history; export weekly summaries or full conversations as PDF.
- **AtaraxBot UI:** A friendly, accessible chat interface for support and explanations.
- **Customizability:** Adjustable tone, CBT technique, sensitivity, and user preferences.

---

## Key Features

- **Offline, On-Device Privacy:** All processing happens locally. No data leaves your device.
- **Contextual CBT:** Dynamic prompt engineering incorporates user memory and recent conversations for highly personalized support.
- **Crisis Detection:** Fine-tuned ONNX classifier flags high-risk entries with confidence scores.
- **Flexible Journaling:** Free-form journaling interface with emotional tagging and CBT technique selection.
- **Emotional & Crisis Analytics:** Token usage, emotion trends, and crisis frequency visualizations.
- **Export Tools:** Download weekly summaries, chat histories, or raw data (JSON/PDF).
- **AtaraxBot:** On-demand, contextual mental health chatbot for quick questions.
- **User-Controlled Memory:** Users can clear or export their data at any time.

---

## How Gemma 3n Is Used

- **Core Model:** AtaraxAI leverages a fine-tuned, quantized version of **Gemma 3n** (2B) running locally with Ollama or Unsloth.
- **Dynamic Prompt Engineering:** User journal entries are combined with session memory, emotion tags, tone, CBT technique, and trauma history to generate adaptive prompts for Gemma 3n, producing rich, context-aware responses.
- **Session Memory:** Summarization of previous entries using T5-small allows for long-term context continuity within token limits.
- **CBT Guidance:** Gemma 3n is instructed via prompt templates to deliver CBT-based reflection, coping strategies, and open-ended questions.
- **Multimodal Readiness:** Architecture supports potential multimodal inputs for future Gemma 3n releases (e.g., voice journaling, image emotion analysis).

---

## Dataset & Fine-tuning

- **CBT Dataset:** Custom dataset of user journal entries, annotated emotions, and ideal CBT responses was created for supervised fine-tuning.
- **Crisis Detection Dataset:** Real-world and synthetic mental health journal entries labeled for crisis risk, fine-tuned on a compact transformer and exported to ONNX for real-time, offline inference.
- **Privacy-Centric:** All datasets were stripped of personal information; no user data leaves the device.
- **Fine-Tuning Pipeline:** Utilized Unsloth and HuggingFace Transformers for efficient, low-resource model training and quantization.

---

## Technical Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/) for fast, responsive, and accessible UI.
- **LLM Inference:** [Ollama](https://ollama.com/) or [Unsloth](https://unsloth.ai/) for local running of quantized Gemma 3n models.
- **CBT & Prompt Generation:** Custom Python modules (`ollama_prompt.py`) for modular prompt engineering and context-aware LLM interaction.
- **Crisis Detection:** ONNX Runtime for lightweight, real-time crisis risk prediction.
- **PDF/Export:** [FPDF](https://pyfpdf.github.io/) for rich Unicode PDF export.
- **Analytics & Visualization:** [Plotly](https://plotly.com/python/) and [Pandas](https://pandas.pydata.org/) for interactive analytics.
- **Session Memory:** JSON-based storage with summarization (T5-small).
- **Custom Chatbot UI:** AtaraxBot floating assistant with base64-encoded custom icon.

---

## **Requirements**  
# Supports: Streamlit Cloud (online) & Local Inference (offline)
# -------------------------------
# Core Streamlit App (Both Online + Offline)
# -------------------------------
streamlit==1.35.0
fpdf==1.7.2
pandas==2.2.2
plotly==5.22.0

# -------------------------------
# NLP Models & Transformers (Both)
# -------------------------------
transformers==4.41.2
torch==2.3.0
sentencepiece  # Required for T5, BERT
scikit-learn==1.5.0
protobuf==4.25.3  # Required for ONNX/transformers compatibility

# -------------------------------
# ONNX Runtime for Crisis Detection (Both)
# -------------------------------
onnxruntime==1.17.1

# -------------------------------
# 🔁 Optional (Only Needed for LOCAL Deployment)
# -------------------------------

# --- Local LLM Inference via Ollama ---
ollama  # Only works locally, not on Streamlit Cloud

# --- Fine-tuning / Fast Inference Tools (if still used) ---
unsloth==0.2.0  # Used for fine-tuning (optional if inference-only)
accelerate==0.30.1  # Required by unsloth for fast training

# --- Local Plotting for Notebook Testing (Optional) ---
matplotlib==3.9.0

---

## Target Impact

- **Empowerment:** Provides anyone, anywhere, with private, stigma-free access to AI-guided CBT and crisis detection.
- **Access:** Designed for low-resource, high-stigma, or connectivity-limited environments (e.g., developing countries, rural areas).
- **Privacy:** Ensures all user data stays local, fostering trust and reducing risk.
- **Personal Growth:** Encourages regular self-reflection and resilience-building, with measurable analytics for progress tracking.
- **Open Source:** Enables community-led improvements, transparency, and responsible AI development in mental health tech.

---

## Demo & Screenshots

> **[Watch the Demo Video Here](#)** <!-- Add YouTube link after recording -->
>
> ![ataraxai-journal](assets/ataraxai_journal.png)
> ![ataraxai-analytics](assets/ataraxai_analytics.png)

---

## Setup & Installation

### 1. **Clone the Repo**
```bash
git clone https://github.com/yourusername/ataraxai.git
cd ataraxai

