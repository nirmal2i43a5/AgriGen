import streamlit as st
from audio_recorder_streamlit import audio_recorder
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.services.speech_service import transcribe_audio
from utils.styles import load_css


def render_voice_modal(whisper_model):
    
    # Load CSS from assets
    load_css('voice_input.css')
    
    # Render waveform animation
    st.markdown("""
    <div class="recording-text">Ask...</div>
    <div class="waveform-container">
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Audio recorder with larger size for modal
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#2ecc71",
        icon_name="microphone",
        icon_size="2x",
        key="voice_modal_recorder"
    )
    
    if audio_bytes:
        if 'last_audio_bytes' not in st.session_state:
            st.session_state.last_audio_bytes = None
        
        if audio_bytes != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            
            with st.spinner(f"Transcribing with {whisper_model}..."):
                try:
                    transcription = transcribe_audio(audio_bytes, whisper_model)
                    
                    if transcription:
                        st.success(f"You said: **{transcription}**")
                        
                        # Set auto-submit flags
                        st.session_state.voice_query = transcription
                        st.session_state.voice_auto_submit = True
                        
                        st.info("Submitting your question...")
                        
                        # Close modal and trigger query
                        return True
                    else:
                        st.warning("No speech detected. Please try again.")
                        return False
                
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
                    if st.button("Try Again"):
                        st.session_state.last_audio_bytes = None
                        st.rerun()
                    return False
    
    return False


def get_voice_query():
    return st.session_state.get('voice_transcription', '')


def clear_voice_query():
    if 'voice_transcription' in st.session_state:
        del st.session_state.voice_transcription
    if 'last_audio_bytes' in st.session_state:
        del st.session_state.last_audio_bytes

