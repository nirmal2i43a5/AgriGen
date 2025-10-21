
import streamlit as st
from typing import Dict, Any, List
import sys
import os
from streamlit_chat import message
from PIL import Image
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.llm.groq_model import get_model_display_name
from utils.styles import load_css


def render_image_preview():
    """Renders image preview above chat input - call this separately before chat messages"""
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    if st.session_state.uploaded_image:
        st.markdown("---")
        col_preview1, col_preview2 = st.columns([3, 1])
        
        with col_preview1:
            st.markdown("**Image attached:**")
            try:
                image = Image.open(st.session_state.uploaded_image)
                st.image(image, width=200)
            except:
                st.warning("Unable to preview image")
        
        with col_preview2:
            if st.button("Remove", key="remove_image"):
                st.session_state.uploaded_image = None
                st.rerun()
        st.markdown("---")


def render_chat_input(whisper_model=None):
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Load CSS from assets
    load_css('chat_interface.css')

    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""

    with st.container():
        col1, col2, col3 = st.columns([1, 9, 1])
        
        with col1:
            uploaded_image = st.file_uploader(
                "Upload",
                type=["png", "jpg", "jpeg", "webp"],
                key="direct_image_upload",
                label_visibility="collapsed",
                help="Upload an image"
            )
            
            if uploaded_image and uploaded_image != st.session_state.uploaded_image:
                st.session_state.uploaded_image = uploaded_image
                st.rerun()

        with col2:
            query = st.text_input(
                "Welcome to the AgriAdvisor-Ai-Assistant! Ask a question to get started.", 
                value=st.session_state.user_query, 
                label_visibility="collapsed",
                key="chat_input",
                placeholder="Ask about crops, diseases, treatments..."
            )
        
        with col3:
            voice_clicked = st.button("🎤", help="Record voice", key="voice_btn")
            
    return query, voice_clicked


def render_chat_messages(messages: List[Dict]):
    
    if not messages:
        return

    # Group consecutive bot messages together
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if msg["role"] == "user":
            message(msg["content"], is_user=True, key=f"user_{i}")
            i += 1
        else:
            # Collect all consecutive bot messages
            bot_messages = []
            j = i
            while j < len(messages) and messages[j]["role"] != "user":
                bot_messages.append((j, messages[j]))
                j += 1
            
            # If multiple bot messages, use tabs
            if len(bot_messages) > 1:
                tab_labels = [get_model_display_name(msg.get("model", "assistant")) for _, msg in bot_messages]
                tabs = st.tabs(tab_labels)
                
                for tab_idx, (tab, (msg_idx, bot_msg)) in enumerate(zip(tabs, bot_messages)):
                    with tab:
                        message(bot_msg["content"], key=f"bot_{msg_idx}", avatar_style="bottts")
                        
                        if bot_msg.get("sources"):
                            with st.expander("View Sources"):
                                for idx, source in enumerate(bot_msg["sources"], 1):
                                    source_path = source.get('source', 'N/A')
                                    filename = os.path.basename(source_path) if source_path != 'N/A' else 'Unknown Source'
                                    st.caption(f'{filename}')
            else:
                # Single bot message - render normally
                msg_idx, bot_msg = bot_messages[0]
                model_name = get_model_display_name(bot_msg.get("model", "assistant"))
                
                with st.container():
                    st.caption(f"Responded with {model_name}")
                    message(bot_msg["content"], key=f"bot_{msg_idx}", avatar_style="bottts")
                    
                    if bot_msg.get("sources"):
                        with st.expander("View Sources"):
                            for idx, source in enumerate(bot_msg["sources"], 1):
                                source_path = source.get('source', 'N/A')
                                filename = os.path.basename(source_path) if source_path != 'N/A' else 'Unknown Source'
                                st.caption(f'{filename}')
            
            i = j
                            


def render_vision_result(answer: str, vision_model: str):
    model_name = get_model_display_name(vision_model, "vision")
    st.markdown(f"# Analysis from {model_name}")
    st.info(answer)