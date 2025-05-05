from faster_whisper import WhisperModel
import sounddevice as sd
import soundfile as sf
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Constants
DURATION = 5
SAMPLE_RATE = 16000
OUTPUT_WAV = "input.wav"

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_sleep_coach_model"
tokenizer_path = "./fine_tuned_sleep_coach_tokenizer"

print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set up text generation pipeline
'''generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

prompt = "Question: what is the average hours someone should sleep?\nAnswer:"
outputs = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

print(outputs[0]["generated_text"])'''

# Record Microphone Input

def record_audio(filename=OUTPUT_WAV, duration=DURATION, rate=SAMPLE_RATE):
    print("Recording for", duration, "seconds...")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1)
    sd.wait()
    sf.write(filename, audio, rate)
    print("Recording complete.")

# Transcribe Speech to Text
def transcribe_audio(filename):
    whisper = WhisperModel("base", device="cpu")
    segments, info = whisper.transcribe(filename)
    result = " ".join([segment.text for segment in segments])
    print("You said:", result)
    return result

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Generate Response from Fine-Tuned Model
def get_response(user_text):
    prompt = f"Question: {user_text}\nAnswer:"
    print("Generating response...")
    output = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    response = output[0]["generated_text"]
    return response.split("Answer:")[-1].strip()

# Speak the Response
def speak_text(text):
    print("Speaking response...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()

# Main Voice Interaction Loop
if __name__ == "__main__":
    print(" Fine-Tuned Voice Sleep Coach Ready. Say 'exit' to quit.")
    while True:
        record_audio()
        user_input = transcribe_audio(OUTPUT_WAV)
        if "exit" in user_input.lower():
            print("Exiting. Sleep well!")
            break
        response = get_response(user_input)
        print("Sleep Coach:", response)
        speak_text(response)
