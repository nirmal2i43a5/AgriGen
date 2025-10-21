import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.llm.groq_model import (
    GROQ_TEXT_MODELS,
    GROQ_VISION_MODELS,
    GROQ_WHISPER_MODELS,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_WHISPER_MODEL
)

def render_header_controls():
    
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        selected_text_models = st.multiselect(
            "Select Text Models",
            options=list(GROQ_TEXT_MODELS.keys()),
            default=[DEFAULT_TEXT_MODEL],
            format_func=lambda x: GROQ_TEXT_MODELS[x],
            help="Select one or more models to compare answers"
        )
    
    with c2:
        selected_vision_model = st.selectbox(
            "Vision Model",
            options=list(GROQ_VISION_MODELS.keys()),
            index=list(GROQ_VISION_MODELS.keys()).index(DEFAULT_VISION_MODEL),
            format_func=lambda x: GROQ_VISION_MODELS[x],
            help="Model for analyzing images"
        )
    
    with c3:
        selected_whisper_model = st.selectbox(
            "Speech Model",
            options=list(GROQ_WHISPER_MODELS.keys()),
            index=list(GROQ_WHISPER_MODELS.keys()).index(DEFAULT_WHISPER_MODEL),
            format_func=lambda x: GROQ_WHISPER_MODELS[x],
            help="Model for voice transcription"
        )
        
    return {
        "text_models": selected_text_models,
        "vision_model": selected_vision_model,
        "whisper_model": selected_whisper_model,
    }
