

import streamlit as st
import os
from pathlib import Path


def load_css(css_file: str):
 
    try:
        # Get the frontend directory path
        frontend_dir = Path(__file__).parent.parent
        css_path = frontend_dir / "assets" / "css" / css_file
        
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {css_file}")
    except Exception as e:
        st.error(f"Error loading CSS {css_file}: {str(e)}")


def load_multiple_css(css_files: list):

    for css_file in css_files:
        load_css(css_file)

