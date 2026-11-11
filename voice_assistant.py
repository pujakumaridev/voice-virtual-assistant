import os
import tempfile
from time import sleep

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from scipy.io import wavfile

load_dotenv()

SAMPLE_RATE = 16000
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = "eleven_multilingual_v2"
STT_MODEL = "scribe_v1"
LLM_MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies are spoken aloud, so keep "
    "them "
    "short (1-3 sentences), conversational, and free of markdown or lists."
    "always respond in the same language as the user."
)

eleven = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

history = []  # conversation memory: list of {"role": ..., "content": ...}


def record() -> str:
    """Record from the mic until the user presses Enter. Returns a WAV file path."""
    input("\nPress Enter to start talking...")
    chunks = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        input("Listening... press Enter to stop.")

    audio = np.concatenate(chunks, axis=0)
    path = os.path.join(tempfile.gettempdir(), "user_input.wav")
    wavfile.write(path, SAMPLE_RATE, audio)
    return path


def transcribe(path: str) -> str:
    """Speech -> text using ElevenLabs Scribe."""
    with open(path, "rb") as f:
        result = eleven.speech_to_text.convert(file=f, model_id=STT_MODEL)
    return result.text.strip()

def think(user_text: str) -> str:
    """Text -> reply, keeping the whole conversation as context."""
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    for attempt in range(4):
        try:
            response = gemini.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=600,
                ),
            )
            break
        except Exception as e:
            if attempt == 3:
                raise
            print("Busy, retrying...")
            sleep(3)
    reply = response.text.strip()
    history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
    return reply


def speak(text: str) -> None:
    """Text -> speech using ElevenLabs TTS, then play it."""
    audio = eleven.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=TTS_MODEL,
        output_format="mp3_44100_128",
    )
    play(audio)


def main() -> None:
    print("Voice assistant ready. Say 'goodbye' to quit, or press Ctrl+C.")
    speak("Hi! How can I help you?")

    while True:
        wav_path = record()
        user_text = transcribe(wav_path)
        if not user_text:
            print("(didn't catch that)")
            continue
        print(f"You: {user_text}")

        if "goodbye" in user_text.lower():
            speak("Goodbye!")
            break

        reply = think(user_text)
        print(f"Assistant: {reply}")
        speak(reply)


if __name__ == "__main__":
    main()