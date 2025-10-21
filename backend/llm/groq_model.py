# Groq Model Configuration and Client Management
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Groq Free Tier Models (2025)
GROQ_TEXT_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile",
    "openai/gpt-oss-120b": "GPT OSS 120B",
    "openai/gpt-oss-20b": "GPT OSS 20B",
    "moonshotai/kimi-k2-instruct-0905": "Kimi K2",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
}

GROQ_VISION_MODELS = {
    "llama-3.2-11b-vision-preview": "Llama 3.2 11B Vision",
    "llama-3.2-90b-vision-preview": "Llama 3.2 90B Vision",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick (Vision)"
}

GROQ_WHISPER_MODELS = {
    "whisper-large-v3": "Whisper Large v3",
    "whisper-large-v3-turbo": "Whisper Large v3 Turbo"
}

GROQ_TTS_MODEL = "playai-tts"

# Default selections
DEFAULT_TEXT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
DEFAULT_WHISPER_MODEL = "whisper-large-v3"


def get_groq_client():
    # Get initialized Groq client
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable not set")
    return Groq(api_key=api_key)


def get_model_display_name(model_id, model_type="text"):
    # Get friendly display name for a model ID
    if model_type == "text":
        return GROQ_TEXT_MODELS.get(model_id, model_id)
    elif model_type == "vision":
        return GROQ_VISION_MODELS.get(model_id, model_id)
    elif model_type == "whisper":
        return GROQ_WHISPER_MODELS.get(model_id, model_id)
    return model_id

