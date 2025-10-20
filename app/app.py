import warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')

import streamlit as st
import os
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import initialize_pipeline, process_documents, ask_question

st.title("Crop Advisory Assistant")

if 'pipeline_initialized' not in st.session_state:
    with st.spinner("Initializing AI system..."):
        st.session_state.qa_chain, st.session_state.vector_store = initialize_pipeline()
        st.session_state.pipeline_initialized = True
    st.success("System ready!")

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type="pdf", 
    accept_multiple_files=True,
    help="Upload agricultural documents to add to the knowledge base"
)

if uploaded_files:
    current_file_names = {f.name for f in uploaded_files}
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in new_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Processing {len(new_files)} new document(s)..."):
                num_chunks = process_documents(temp_dir, st.session_state.vector_store)
                st.success(f"{len(new_files)} document(s) processed successfully! ({num_chunks} chunks added)")
            
            st.session_state.processed_files.update({f.name for f in new_files})
    else:
        st.info(f"{len(uploaded_files)} document(s) already processed")

question = st.text_input(
    placeholder="e.g., What are the best practices for wheat cultivation?"
)

if question:
    with st.spinner("Finding answer..."):
        answer = ask_question(st.session_state.qa_chain, question)
        st.write("### Answer")
        st.write(answer)

# with st.sidebar:
#     st.header("System Status")
#     st.write(f"**Documents processed:** {len(st.session_state.processed_files)}")
#     if st.session_state.processed_files:
#         st.write("**Files:**")
#         for filename in st.session_state.processed_files:
#             st.write(f"- {filename}")
    
    # if st.button("Clear All Documents"):
    #     st.session_state.processed_files = set()
    #     st.session_state.qa_chain, st.session_state.vector_store = initialize_pipeline()
    #     st.success("All documents cleared!")
    #     st.rerun()