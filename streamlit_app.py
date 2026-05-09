from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Music Evolution Graph", layout="wide")
st.markdown(
    """
    <style>
      header, footer, #MainMenu {visibility: hidden;}
      .block-container {padding: 0; max-width: 100%;}
      iframe {height: 95vh !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
components.html(html, height=900, scrolling=False)
