import streamlit as st
from audio_recorder_streamlit import audio_recorder
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from app.services.speech_service import transcribe_audio


def render_voice_modal(whisper_model):
    
    st.markdown("""
    <style>
        .waveform-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 3px;
            height: 50px;
            margin: 20px 0;
        }
        .waveform-bar {
            width: 3px;
            background: linear-gradient(180deg, #2ecc71 0%, #27ae60 100%);
            border-radius: 2px;
            animation: wave 1.2s ease-in-out infinite;
        }
        .waveform-bar:nth-child(1) { height: 20%; animation-delay: 0s; }
        .waveform-bar:nth-child(2) { height: 40%; animation-delay: 0.1s; }
        .waveform-bar:nth-child(3) { height: 60%; animation-delay: 0.2s; }
        .waveform-bar:nth-child(4) { height: 80%; animation-delay: 0.3s; }
        .waveform-bar:nth-child(5) { height: 100%; animation-delay: 0.4s; }
        .waveform-bar:nth-child(6) { height: 80%; animation-delay: 0.5s; }
        .waveform-bar:nth-child(7) { height: 60%; animation-delay: 0.6s; }
        .waveform-bar:nth-child(8) { height: 40%; animation-delay: 0.7s; }
        .waveform-bar:nth-child(9) { height: 20%; animation-delay: 0.8s; }
        
        @keyframes wave {
            0%, 100% { transform: scaleY(0.5); opacity: 0.5; }
            50% { transform: scaleY(1); opacity: 1; }
        }
        
        .recording-text {
            text-align: center;
            font-size: 1.2rem;
            color: #2ecc71;
            margin-bottom: 10px;
        }
    </style>
    
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

