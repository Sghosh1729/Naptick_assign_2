from faster_whisper import WhisperModel
import sounddevice as sd
import soundfile as sf
import pyttsx3
from transformers import pipeline

# hyper parameters
DURATION = 5  # seconds to record
SAMPLE_RATE = 16000
OUTPUT_WAV = "input.wav"

# Record Microphone Input
def record_audio(filename=OUTPUT_WAV, duration=DURATION, rate=SAMPLE_RATE):
    print("Recording for", duration, "seconds...")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=1)
    sd.wait()
    sf.write(filename, audio, rate)
    print("Recording complete.")

#Transcribe Speech to Text
def transcribe_audio(filename):
    model = WhisperModel("base", device="cpu")
    segments, info = model.transcribe(filename)
    result = " ".join([segment.text for segment in segments])
    print("You said:", result)
    return result

#Generate Text Response using LLM
def get_response(user_text):
    print("Generating response...")
    generator = pipeline("text-generation", model="distilgpt2", device_map="auto")
    prompt = f"You are a helpful sleep coach. Respond clearly to: {user_text}"
    output = generator(prompt, max_new_tokens=100)[0]["generated_text"]
    # Trim to remove repetition
    return output.split("Respond clearly to:")[-1].strip()

# Speak the Response
def speak_text(text):
    print("Speaking response...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()

#Voice-to-Voice Loop
if __name__ == "__main__":
    print(" Voice Sleep Coach Ready. Say 'exit' to quit.")
    while True:
        record_audio()
        user_input = transcribe_audio(OUTPUT_WAV)
        if "exit" in user_input.lower():
            print("Exiting. Sleep well!")
            break
        response = get_response(user_input)
        print("Sleep Coach:", response)
        speak_text(response)
