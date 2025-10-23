import io
from typing import Optional
from backend.src.llm.groq_model import get_groq_client


def transcribe_audio(audio_bytes, whisper_model="whisper-large-v3"):
   
    try:
        client = get_groq_client()
        
        print(f" Transcribing audio with {whisper_model}...")
        
   
   
        response = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model=whisper_model,
            temperature=0.0,  
            response_format="verbose_json" 
        )
        
        # Extract transcription text
        transcription = response.text if hasattr(response, 'text') else str(response)
        print(f" Transcription complete: {len(transcription)} characters")
        
        
        if hasattr(response, 'language'):
            print(f" Detected language: {response.language}")
        if hasattr(response, 'duration'):
            print(f" Audio duration: {response.duration:.2f}s")
        
        return transcription.strip()
    
    except Exception as e:
        print(f" Audio transcription failed: {e}")
        raise


def text_to_speech(text, model="playai-tts"):

    try:
        client = get_groq_client()
        
        print(f" Generating speech with {model}...")
    
    
        response = client.audio.speech.create(
            model=model,
            input=text
        )
        
        return response.content
    
    except Exception as e:
        print(f"[WARNING] TTS not available or failed: {e}")
        return None

