import os
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import asyncio

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(api_key=api_key)


class Speaker:
    def __init__(
        self, voice_id: str = "JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2"
    ):
        self.voice_id = voice_id
        self.model_id = model_id

    def speak(self, text: str) -> None:
        audio_stream = client.text_to_speech.convert_as_stream(
            text=text, voice_id=self.voice_id, model_id=self.model_id
        )
        stream(audio_stream)

    async def speak_async(self, text: str) -> None:
        await asyncio.to_thread(self.speak, text)
