import streamlit as st
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from main import create_pipeline

st.set_page_config(page_title="RAG Pipeline", page_icon="🔍")
st.title("🔍 RAG Pipeline")
st.write("Faça perguntas sobre os documentos indexados.")

@st.cache_resource
def load_pipeline():
    return create_pipeline()

pipeline = load_pipeline()

question = st.text_input("Sua pergunta:")
if st.button("Perguntar") and question:
    with st.spinner("Buscando..."):
        answer = pipeline.process_query(question)
    st.success(answer)
