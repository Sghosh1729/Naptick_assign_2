# Naptick_assign_2

##  Overview

This project aims to develop a **voice-interactive AI sleep coach** that provides personalized, evidence-backed sleep advice using large language models (LLMs). The system:
- Takes **voice input** from users,
- Processes it through **automatic speech recognition (ASR)**,
- Generates natural language responses using a **fine-tuned model**, and
- Delivers spoken feedback via **text-to-speech (TTS)**.

##  Core Features

- Voice-to-voice interaction loop  
- Personalized responses based on user input  
  Integrated sleep science knowledge from datasets and literature  

##  Technical Stack

- `Python` for pipeline orchestration  
- `Hugging Face Transformers` for model loading and training  
- `Kaggle Datasets` for real-world sleep data  
- `Faster-Whisper` for ASR  
- `pyttsx3` for speech synthesis  
- `sounddevice` and `soundfile` for real-time audio recording  

##  Base Model

The assistant is built using:
- `faster-whisper` for speech recognition  
- `distilgpt2` for generating responses  
- `pyttsx3` for text-to-speech conversion  

It allows natural voice interactions where users can ask questions or discuss topics related to sleep and well-being.

##  Key Functionalities

- **Voice Input Recording**: Captures user voice using a microphone for 5 seconds  
- **Speech-to-Text (STT)**: Transcribes voice input using Whisper base model  
- **LLM Response Generation**: Uses distilgpt2 to generate relevant replies  
- **Text-to-Speech (TTS)**: Converts model output to speech using pyttsx3  

> This base model serves as a prototype for more advanced voice-based assistants, especially in health coaching domains like sleep.

---

##  Data & Training Strategy

To make the model relevant to sleep topics, a hybrid fine-tuning strategy was applied:

###  Datasets Used

- `sleep_health_lifestyle`  
- `sleep_data`  
- `sleep_efficiency`  
- `student_sleep_patterns`  

These datasets were used for adapting and fine-tuning the base model.

###  Fine-tuning Data Sources

- **Curated QA Pairs**: From public sleep datasets  
- **Instructional Examples**: Based on structured sleep metrics (e.g., age, duration, stress, caffeine)  
- **Clinical Insight Prompts**: Informed by research-backed data  

---

##  Finetuning & Adaptation Pipeline

- **Data Loading**: JSON-based QA pairs (Question/Answer format)  
- **Text Formatting**: Into prompt style — `Question: ... Answer: ...`  
- **Tokenization**: Using `AutoTokenizer` (max 512 tokens)  
- **Model Setup**: Configured `distilgpt2` for causal language modeling  
- **Trainer Setup**: Uses `Trainer` and `TrainingArguments` from Hugging Face  
- **Training Loop**: 3 epochs with checkpoint saving  
- **Model Saving**: Stores model and tokenizer for inference use  

This enables the creation of a **compact**, **domain-specific**, and **responsive** model tailored to sleep-related interactions.

---

##  Improved Model Capabilities

With fine-tuning, the AI coach becomes:

-  **Domain-Aware**: Understands sleep-related questions in depth  
-  **Voice-Interactive**: Handles voice input/output seamlessly  
-  **GPU-Optimized**: Uses CUDA automatically if available  
-  **Controlled Generation**: Uses `temperature=0.7` for creative yet relevant responses  

###  Example Use Case

> **Q**: “How much should a teenager sleep?”  
> **A**: *(spoken reply)* “Teenagers typically need 8 to 10 hours of sleep per night for optimal health and performance.”

---

##  Outcome

The result is a **deployable AI assistant** that can answer:
- “How much sleep should I get if I’m a stressed college student?”
- “What’s the impact of caffeine on my sleep quality?”

This hybrid pipeline enhances contextual reasoning about human sleep behavior, increasing both **generalization** and **real-world reliability**.

---



