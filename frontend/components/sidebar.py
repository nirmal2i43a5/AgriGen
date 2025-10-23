import streamlit as st
import sys
import os
import tempfile
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.services.chat_memory import ChatMemory
from utils.styles import load_css

def render_chat_sidebar(memory: ChatMemory):
    with st.sidebar:
    
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
            handle_pdf_upload(uploaded_pdfs)
              
        st.markdown("### Chats")
        
        sessions = memory.get_all_sessions()
        
        if not sessions:
            st.caption("No past chats yet.")
        else:
            for session in sessions:
                # Create a container for each chat item
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        # Truncate long names
                        display_name = session["name"]
                        if len(display_name) > 30:
                            display_name = display_name[:27] + "..."
                        
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
                    
                    with col2:
                        # Three-dot menu button
                        if st.button("⋯", key=f"menu_{session['id']}", help="Chat options"):
                            # Close any other open menus first
                            for other_session in sessions:
                                if other_session['id'] != session['id']:
                                    st.session_state[f"show_menu_{other_session['id']}"] = False
                            st.session_state[f"show_menu_{session['id']}"] = not st.session_state.get(f"show_menu_{session['id']}", False)
                            st.rerun()
                    
                    if st.session_state.get(f"show_menu_{session['id']}", False):
                        st.markdown(f"""
                        <div class="chatgpt-dropdown-popup" id="dropdown_{session['id']}">
                            <div class="dropdown-item" onclick="document.querySelector('[data-testid=\\"stButton\\"] button[key=\\"popup_rename_{session['id']}\\"]').click()">
                                Rename
                            </div>
                            <div class="dropdown-item delete-item" onclick="document.querySelector('[data-testid=\\"stButton\\"] button[key=\\"popup_delete_{session['id']}\\"]').click()">
                                Delete
                            </div>
                        </div>
                        <script>
                            setTimeout(function() {{
                                var menuButton = document.querySelector('[key=\\"menu_{session['id']}\\"]');
                                if (menuButton) {{
                                    var rect = menuButton.getBoundingClientRect();
                                    var dropdown = document.querySelector('#dropdown_{session['id']}');
                                    if (dropdown) {{
                                        dropdown.style.left = (rect.right + 50) + 'px';
                                        dropdown.style.top = rect.top + 'px';
                                        dropdown.style.position = 'fixed';
                                    }}
                                }}
                            }}, 100);
                        </script>

                        """, unsafe_allow_html=True)
                        
                        # Using session state to handle clicks from popup
                        if st.session_state.get(f"popup_rename_{session['id']}", False):
                            st.session_state[f"rename_mode_{session['id']}"] = True
                            st.session_state[f"show_menu_{session['id']}"] = False
                            st.session_state[f"popup_rename_{session['id']}"] = False
                            st.rerun()
                        
                        if st.session_state.get(f"popup_delete_{session['id']}", False):
                            st.session_state[f"delete_confirm_{session['id']}"] = True
                            st.session_state[f"show_menu_{session['id']}"] = False
                            st.session_state[f"popup_delete_{session['id']}"] = False
                            st.rerun()
                    
                    if st.session_state.get(f"rename_mode_{session['id']}", False):
                        new_name = st.text_input(
                            "New name:", 
                            value=session["name"],
                            key=f"rename_input_{session['id']}"
                        )
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("Save", key=f"save_rename_{session['id']}"):
                                if new_name and new_name.strip():
                                    memory.rename_session(session['id'], new_name.strip())
                                    st.session_state[f"rename_mode_{session['id']}"] = False
                                    st.rerun()
                        
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_rename_{session['id']}"):
                                st.session_state[f"rename_mode_{session['id']}"] = False
                                st.rerun()
                    
                    if st.session_state.get(f"delete_confirm_{session['id']}", False):
                        st.warning(f"Delete '{session['name']}'?")
                        
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("Yes, Delete", key=f"confirm_delete_{session['id']}", type="primary"):
                                memory.delete_session(session['id'])
                                if st.session_state.get('session_id') == session['id']:
                                    st.session_state.session_id = None
                                    st.session_state.messages = []
                                st.session_state[f"delete_confirm_{session['id']}"] = False
                                st.rerun()
                        
                        with col_no:
                            if st.button("Cancel", key=f"cancel_delete_{session['id']}"):
                                st.session_state[f"delete_confirm_{session['id']}"] = False
                                st.rerun()
              


def handle_pdf_upload(uploaded_pdfs):
  
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    new_files = [f for f in uploaded_pdfs if f.name not in st.session_state.processed_files]
    
    if new_files:
        with st.spinner(f"Processing {len(new_files)} document(s)..."):
            try:
                files = []
                for uploaded_file in new_files:
                    files.append(('files', (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')))
                
                endpoint = "http://localhost:8000/api/documents/upload/admin"
                
                response = requests.post(endpoint, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"{result['message']} ({result['chunks_added']} chunks)")
                    st.info(" Saved to permanent knowledge base")
                    st.session_state.processed_files.update({f.name for f in new_files})
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend API. Please start the backend server.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(f"Error: {str(e)}")