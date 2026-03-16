import streamlit as st
import json
import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# IMPORTS PRIMEIRO
from main import create_pipeline, get_files_in_directory

st.set_page_config(page_title="RAG Pipeline", page_icon="🔍")
st.title("🔍 RAG Pipeline - IF")
st.write("Faça perguntas sobre os documentos indexados.")

@st.cache_resource  
def load_pipeline():
    pipeline = create_pipeline()
    
    # ✅ AUTO-REINDEX AQUI (pipeline existe!)
    db_path = "data/sample-lancedb/rag-table.lance"
    if not os.path.exists(db_path):
        st.info("🗑️ First run - indexing documents...")
        docs = get_files_in_directory("sample_data/source/")
        pipeline.add_documents(docs)
        st.success("✅ Documents indexed!")
    
    return pipeline

pipeline = load_pipeline()

question = st.text_input("Sua pergunta:")
if st.button("Perguntar") and question:
    with st.spinner("Buscando..."):
        answer = pipeline.process_query(question)
    st.markdown(answer)
