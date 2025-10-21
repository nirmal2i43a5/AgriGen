# Speech Services: Speech-to-Text and Text-to-Speech using Groq
import io
from typing import Optional
from backend.llm.groq_model import get_groq_client


def transcribe_audio(audio_bytes, whisper_model="whisper-large-v3"):
    # Transcribe audio to text using Groq Whisper
    try:
        client = get_groq_client()
        
        print(f"[INFO] Transcribing audio with {whisper_model}...")
        
        # Use Groq's recommended format with tuple (filename, bytes)
        response = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model=whisper_model,
            temperature=0.0,  # More deterministic for consistent results
            response_format="verbose_json"  # Get detailed response
        )
        
        # Extract transcription text
        transcription = response.text if hasattr(response, 'text') else str(response)
        print(f"[INFO] Transcription complete: {len(transcription)} characters")
        
        # Log additional info if available
        if hasattr(response, 'language'):
            print(f"[INFO] Detected language: {response.language}")
        if hasattr(response, 'duration'):
            print(f"[INFO] Audio duration: {response.duration:.2f}s")
        
        return transcription.strip()
    
    except Exception as e:
        print(f"[ERROR] Audio transcription failed: {e}")
        raise


def text_to_speech(text, model="playai-tts"):
    # Convert text to speech using Groq TTS (optional feature)
    try:
        client = get_groq_client()
        
        print(f"[INFO] Generating speech with {model}...")
        
        # Note: Groq TTS API may have different interface
        # This is a placeholder for future implementation
        response = client.audio.speech.create(
            model=model,
            input=text
        )
        
        return response.content
    
    except Exception as e:
        print(f"[WARNING] TTS not available or failed: {e}")
        return None

