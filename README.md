# Voice Virtual Assistant

A voice-enabled AI assistant built in Python. You speak, it listens, thinks, and answers out loud.

## How it works
1. Microphone input is recorded.
2. ElevenLabs Scribe converts speech to text.
3. Gemini generates a short reply, remembering the conversation.
4. ElevenLabs text-to-speech speaks the reply.

## Setup
1. Install Python and ffmpeg.
2. Install packages: `pip install -r requirements.txt`
3. Create a `.env` file:
   ELEVENLABS_API_KEY=your_key
   GEMINI_API_KEY=your_key
   ```
4. Run: `py voice_assistant.py`

## Usage
Press Enter, speak, press Enter again. Say "goodbye" to quit.

## Tech
Python, ElevenLabs API, Google Gemini API, sounddevice
