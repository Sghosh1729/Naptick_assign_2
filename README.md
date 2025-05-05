# Naptick_assign_2
Overview:
This project aims to develop a voice-interactive AI sleep coach that provides personalized, evidence-backed sleep advice using large language models (LLMs). The system takes voice input from users, processes it through automatic speech recognition (ASR), generates natural language responses using a fine-tuned model, and delivers spoken feedback through text-to-speech (TTS).
Core Features:
•	Voice-to-voice interaction loop
•	Personalized responses based on user input
•	Integrated sleep science knowledge from datasets and literature

Technical Stack:
•	Python for pipeline orchestration
•	Hugging Face Transformers for model loading and training
•	Kaggle Datasets for real-world sleep data
•	Faster-Whisper for ASR
•	pyttsx3 for speech synthesis
•	sounddevice and soundfile for real-time audio recording.

Base Model:
The model is a voice-to-voice AI sleep coach built using the faster_whisper speech recognition model, distilgpt2 for response generation, and pyttsx3 for speech synthesis. It allows users to interact naturally via voice to ask questions or discuss topics related to sleep and well-being.
 Key Features:
•	Voice Input Recording: Captures user queries using the microphone for 5 seconds.
•	Speech-to-Text (STT): Uses the lightweight Whisper model (base variant) to transcribe spoken input into text.
•	LLM Response Generation: Passes the transcription to a distilled GPT-2 model that acts as a helpful sleep coach and generates a relevant response.
•	Text-to-Speech (TTS): Converts the generated text back into speech using pyttsx3, enabling a seamless voice conversation.

This base model serves as a prototype for more advanced voice-based assistants, particularly in health coaching domains like sleep improvement.

Data & Training Strategy:
To enhance the model’s relevance to sleep-related questions, a hybrid fine-tuning approach was used:
•	Datasets Used: Kaggle datasets 
  1.	sleep_health_lifestyle
  2.	sleep_data
  3.	sleep_efficiency
  4.	student_sleep_patterns
were used for adaptation and fine tuning of the base model 


•	Curated QA Pairs: Derived from public sleep datasets (e.g., student sleep patterns, lifestyle and efficiency data).
•	Instructional Examples: Generated from structured sleep metrics (age, duration, stress, caffeine intake, etc.).
•	Clinical Insight Prompts: Formulated using research-backed data.
Finetuning and Adaptation:
The lightweight language model (distilgpt2) was finetuned using custom question-answer pairs about sleep extracted from real-world datasets. The goal is to create a domain-specific conversational model that can serve as a knowledgeable and personalized AI sleep coach.
 Key Steps:
•	Data Loading: Loads QA pairs from a JSON file, each consisting of a question about sleep and a corresponding answer.
•	Text Formatting: Converts the data into a prompt style ("Question: ... Answer: ...") suitable for language modeling.
•	Tokenization: Prepares the data using AutoTokenizer, ensuring input length fits within a manageable limit (e.g., 512 tokens).
•	Model Setup: Loads distilgpt2 and adapts it for fine-tuning with padding and label configuration.
•	Trainer Configuration: Uses Hugging Face’s Trainer class with custom TrainingArguments to train the model on a GPU/CPU.
•	Training Loop: Runs for 3 epochs and saves model checkpoints periodically.
•	Model Saving: Stores the fine-tuned model and tokenizer for later use in applications like voice-based sleep assistants.
This code enables building a compact, responsive model that can provide more relevant and context-aware responses related to sleep health

Improved Model:
The base voice-to-voice AI sleep coach was improved using a fine-tuned language model. It combines speech recognition, custom language generation, and text-to-speech to deliver a seamless conversational experience tailored to sleep-related queries.
 Key Features:
•	Fine-Tuned GPT Model: Loads a custom-trained distilgpt2 model specialized on sleep health QA data, providing more accurate and domain-specific responses.
•	Speech-to-Text: Uses faster-whisper to transcribe user input in real-time from microphone recordings.
•	Text Generation Pipeline: Generates natural and helpful answers to sleep-related questions using the fine-tuned model, with controlled creativity (temperature=0.7) for engaging replies.
•	Voice Output: Speaks the response aloud using pyttsx3, completing the voice-to-voice interaction loop.
•	GPU-Ready: Automatically uses CUDA if available, enhancing performance during inference.
 Example Use Case:
Ask the assistant: “How much should a teenager sleep?”
It replies with a spoken, personalized response based on your custom sleep QA dataset.

Outcome:
A deployable AI assistant that can answer user questions like:
“How much sleep should I get if I’m a stressed college student?”
“What’s the impact of caffeine on my sleep quality?”
This hybrid training pipeline helps the model reason contextually about human sleep habits, improving both generalization and reliability in real-world use cases.
