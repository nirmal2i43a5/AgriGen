import streamlit as st
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.services.chat_memory import ChatMemory
from backend.src.pipeline import process_documents
from utils.styles import load_css

def render_chat_sidebar(memory: ChatMemory, vector_store):
    with st.sidebar:
        # Load CSS from assets
        load_css('sidebar.css')

 
        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.last_processed_query = ""
            st.session_state.user_query = ""
            if 'uploaded_image' in st.session_state:
                st.session_state.uploaded_image = None
            st.rerun()
        

        st.markdown("#### Train Knowledge Base")
        uploaded_pdfs = st.file_uploader(
            "Upload PDFs", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="sidebar_pdf_upload",
            label_visibility="collapsed"
        )
        
        if uploaded_pdfs:
            handle_pdf_upload(uploaded_pdfs, vector_store)
              
        st.markdown("### Chat History")
        
        sessions = memory.get_all_sessions()
        
        if not sessions:
            st.caption("No past chats yet.")
        else:
            for session in sessions:
                # Truncate long names
                display_name = session["name"]
                if len(display_name) > 35:
                    display_name = display_name[:32] + "..."
                
                
                if st.button(
                    display_name, 
                    key=f"session_{session['id']}", 
                    use_container_width=True
                ):
                    st.session_state.session_id = session['id']
                    st.session_state.messages = []  
                    st.session_state.last_processed_query = ""
                    st.session_state.user_query = ""
                    st.rerun()
              


def handle_pdf_upload(uploaded_pdfs, vector_store):
   

    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    new_files = [f for f in uploaded_pdfs if f.name not in st.session_state.processed_files]
    
    if new_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in new_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Processing {len(new_files)} document(s)..."):
                try:
                    num_chunks = process_documents(temp_dir, vector_store)
                    st.success(f"{len(new_files)} document(s) processed! ({num_chunks} chunks)")
                    st.session_state.processed_files.update({f.name for f in new_files})
                except Exception as e:
                    st.error(f"Error: {str(e)}")