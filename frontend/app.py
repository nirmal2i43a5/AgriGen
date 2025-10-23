import warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')

import streamlit as st
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.src.llm.model_router import ModelRouter
from backend.services.chat_memory import ChatMemory
from backend.services.image_processor import process_image_question, validate_image
from frontend.components.sidebar import render_chat_sidebar
from frontend.components.header_controls import render_header_controls
from frontend.components.chat_interface import render_chat_input, render_chat_messages, render_image_preview
from streamlit_chat import message

st.set_page_config(
    page_title="AgriGen - Farm Advisor Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    /* ChatGPT-style input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 12px 20px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2ecc71;
        box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.2);
    }
    
    /* Button styling for icons */
    .stButton > button {
        border-radius: 50%;
        padding: 8px;
        height: 40px;
        width: 40px;
    }
    
    /* Send button */
    div[data-testid="column"]:nth-child(4) .stButton > button {
        border-radius: 20px;
        width: 100%;
        height: 40px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="...")
def load_rag_pipeline():
    from backend.src.rag_pipeline import initialize_rag_pipeline
    return initialize_rag_pipeline()


@st.cache_resource(show_spinner="...")
def load_model_router():
    return ModelRouter()

@st.cache_resource(show_spinner="...")
def load_chat_memory():
    return ChatMemory()


def initialize_app():
    
    if 'pipeline_initialized' not in st.session_state:
        
        
        #  with st.spinner("Initializing Your AgriGen - Farm Advisor Assistant..."):
        #     try:
        #         # Initialize with new RAG system
        #         from backend.src.rag_pipeline import initialize_rag_pipeline
        #         st.session_state.rag_pipeline = initialize_rag_pipeline()
        #         st.session_state.model_router = ModelRouter()
        #         st.session_state.chat_memory = ChatMemory()
        #         st.session_state.pipeline_initialized = True
        #     except Exception as e:
        #         st.error(f"Initialization failed: {str(e)}")
        #         st.stop()
        
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading AI models...")
            progress_bar.progress(20)
            
            # Initialize components step by step with caching
            status_text.text("Initializing Content and Knowledge Base. Have Patience...")
            progress_bar.progress(40)
            
            st.session_state.rag_pipeline = load_rag_pipeline()
            # status_text.text("Loading knowledge base...")
            progress_bar.progress(60)
            
            st.session_state.model_router = load_model_router()
            # status_text.text("Setting up model router...")
            progress_bar.progress(80)
            
            st.session_state.chat_memory = load_chat_memory()
            progress_bar.progress(100)
            
            st.session_state.pipeline_initialized = True
            status_text.text("AgriGen is ready!")
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

    if 'voice_auto_submit' not in st.session_state:
        st.session_state.voice_auto_submit = False
    if 'voice_query' not in st.session_state:
        st.session_state.voice_query = ""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_processed_query' not in st.session_state:
        st.session_state.last_processed_query = ""


@st.dialog("Voice Recording")
def voice_modal(whisper_model):
    from audio_recorder_streamlit import audio_recorder
    from backend.services.speech_service import transcribe_audio
    import streamlit.components.v1 as components
    
    # Auto-start flag
    if 'voice_modal_opened' not in st.session_state:
        st.session_state.voice_modal_opened = True
    
    # Waveform animation
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
    
    <div class="recording-text">Listening...</div>
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
    
    # Auto-click the microphone button using JavaScript
    components.html("""
        <script>
            // Wait for the audio recorder to be rendered
            const autoClickMic = () => {
                const iframes = window.parent.document.querySelectorAll('iframe');
                
                // Find the audio recorder button in parent document
                const buttons = window.parent.document.querySelectorAll('button');
                buttons.forEach(button => {
                    // Look for microphone icon or audio recorder button
                    const svg = button.querySelector('svg');
                    if (svg) {
                        const iconPath = svg.querySelector('path');
                        if (iconPath || button.innerText === '') {
                            // Check if button has audio recorder styling
                            const style = window.getComputedStyle(button);
                            if (style.borderRadius.includes('50%') || button.closest('[data-testid="stVerticalBlock"]')) {
                                button.click();
                                console.log('Auto-clicked microphone button');
                            }
                        }
                    }
                });
            };
            
            // Try multiple times to ensure the button is rendered
            setTimeout(autoClickMic, 100);
            setTimeout(autoClickMic, 300);
            setTimeout(autoClickMic, 500);
        </script>
    """, height=0)
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#2ecc71",
        icon_name="microphone",
        icon_size="2x",
        key="voice_modal_recorder"
    )
    
    if audio_bytes:
        with st.spinner("Transcribing..."):
            try:
                transcription = transcribe_audio(audio_bytes, whisper_model)
                
                if transcription:
                    
                    # Set the transcribed text and auto-submit flag
                    st.session_state.voice_query = transcription
                    st.session_state.voice_auto_submit = True
                    
                    # Clean up modal state
                    if 'voice_modal_opened' in st.session_state:
                        del st.session_state.voice_modal_opened
                    
                    st.info("Submitting your question...")
                    st.rerun()
                else:
                    st.warning("No speech detected. Please try again.")
            
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")
    
    if st.button("Cancel", use_container_width=True):
        # Clean up modal state
        if 'voice_modal_opened' in st.session_state:
            del st.session_state.voice_modal_opened
        st.rerun()


def main():
    initialize_app()
    
    render_chat_sidebar(st.session_state.chat_memory)

    if st.session_state.session_id and not st.session_state.messages:
        st.session_state.messages = st.session_state.chat_memory.get_session_history(st.session_state.session_id)

    st.markdown('<div class="main-header">AgriGen - Farm Advisor Assistant</div>', unsafe_allow_html=True)
    
    header_data = render_header_controls()
    selected_text_models = header_data["text_models"]
    selected_vision_model = header_data["vision_model"]
    selected_whisper_model = header_data["whisper_model"]
    
    st.divider()

    render_chat_messages(st.session_state.messages)
    
    render_image_preview()
    
    query, voice_clicked = render_chat_input(selected_whisper_model)
    
    if voice_clicked:
        voice_modal(selected_whisper_model)

    # Check if query exists and has changed from last processed query
    if 'last_processed_query' not in st.session_state:
        st.session_state.last_processed_query = ""
    
    if (query and query != st.session_state.last_processed_query) or st.session_state.get('voice_auto_submit', False):
        if st.session_state.get('voice_auto_submit', False):
            query = st.session_state.voice_query
            st.session_state.voice_auto_submit = False
        
        # Create a new session if this is the first message
        if not st.session_state.session_id:
            session_name = query[:40] # Use first 40 chars of query as session name
            st.session_state.session_id = st.session_state.chat_memory.create_new_session(session_name)

        st.session_state.messages.append({"role": "user", "content": query})
        
        st.session_state.last_processed_query = query
        
        process_query_and_update_chat(
            query=query,
            session_id=st.session_state.session_id,
            selected_text_models=selected_text_models,
            selected_vision_model=selected_vision_model
        )
        
        if 'voice_query' in st.session_state:
            del st.session_state.voice_query
        
        st.session_state.user_query = ""
        st.rerun()


def process_query_and_update_chat(query: str, session_id: int, selected_text_models: list, selected_vision_model: str):
    
    uploaded_image = st.session_state.get('uploaded_image', None)

    if uploaded_image:
        with st.spinner(f"Analyzing image with {selected_vision_model}..."):
            try:
                image_bytes = uploaded_image.read()
                validate_image(image_bytes)
                
                vision_answer = process_image_question(image_bytes, query, selected_vision_model)
                
                # Append to messages and save to DB
                bot_message = {"role": "bot", "model": selected_vision_model, "content": vision_answer, "sources": []}
                st.session_state.messages.append(bot_message)
                st.session_state.chat_memory.save_exchange(session_id, query, selected_vision_model, vision_answer)
                
                st.session_state.uploaded_image = None
                uploaded_image.seek(0)

            except Exception as e:
                st.error(f"Image processing failed: {str(e)}")
        
    # Text processing 
    if selected_text_models:
        with st.spinner(f"Querying {len(selected_text_models)} model(s)..."):
            try:
                # Use the new RAG pipeline retriever
                retriever = st.session_state.rag_pipeline.get_retriever()
                results = st.session_state.model_router.ask_multi_models(
                    models=selected_text_models,
                    query=query,
                    retriever=retriever,
                    top_k=3
                )
                
                # Append each model's response to messages and save to DB
                for model_id, result in results.items():
                    # Add fallback indicator to the content if fallback was used
                    content = result["answer"]
                    if result.get("fallback_used", False):
                        content = f"*General advice (no specific documents found)*\n\n{content}"
                    
                    bot_message = {
                        "role": "bot", 
                        "model": model_id, 
                        "content": content, 
                        "sources": result.get("sources", []),
                        "fallback_used": result.get("fallback_used", False)
                    }
                    st.session_state.messages.append(bot_message)
                    st.session_state.chat_memory.save_exchange(session_id, query, model_id, result["answer"], result.get("sources"))

            except Exception as e:
                st.error(f"Query processing failed: {str(e)}")


if __name__ == "__main__":
    main()
