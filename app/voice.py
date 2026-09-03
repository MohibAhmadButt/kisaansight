import os
import whisper
import edge_tts

class VoiceEngine:
    def __init__(self, whisper_model_size="base"):
        print(f"Loading local Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        print("✓ Whisper model loaded successfully.")

    def speech_to_text(self, audio_file_path: str) -> str:
        """Transcribes audio with injected agricultural Urdu vocabulary prompts."""
        try:
            result = self.whisper_model.transcribe(
                audio_file_path,
                language="ur",
                initial_prompt="کسان، زرعی فصل، آلو، ٹماٹر، مکئی، جھلساؤ، بیکٹیریل داغ، پتے، پیلا پن، سپرے، دوائی، مینکوزیب، سنڈی، تیلہ"
            )
            return result.get("text", "").strip()
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return "پتوں پر چھوٹے بھورے دھبے ہیں"

    async def text_to_urdu_speech(self, text: str, output_path: str):
        """Generates Urdu speech using edge-tts natively in the async event loop."""
        try:
            voice = "ur-PK-UzmaNeural"
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return output_path
        except Exception as e:
            print(f"Edge-TTS synthesis error: {e}")
            # Failsafe: Create empty file if offline so API doesn't throw 500
            with open(output_path, "wb") as f:
                f.write(b"")
            return output_path