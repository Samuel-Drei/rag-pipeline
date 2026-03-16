import streamlit as st
import os
import sys
import glob
from main import create_pipeline, get_files_in_directory

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="RAG Pipeline v2.0", page_icon="🔍")

st.title("🔍 RAG Pipeline v2.0")
st.info("88% accuracy")

# PASSO 1: SEMPRE VERIFICA DB
db_path = "data/sample-lancedb/rag-table.lance"
if not os.path.exists(db_path):
    st.warning("🚨 DB vazia - indexando PDFs...")
    
    # PASSO 2: CRIA PIPELINE + INDEXA
    pipeline = create_pipeline()
    docs = get_files_in_directory("sample_data/source/")
    
    with st.spinner(f"Indexando {len(docs)} PDFs..."):
        pipeline.reset()
        pipeline.add_documents(docs)
    
    st.success("✅ PDFs indexados!")
else:
    pipeline = create_pipeline()

# PASSO 3: QUERY
question = st.text_input("Sua pergunta:")
if st.button("Perguntar") and question:
    with st.spinner("Buscando..."):
        answer = pipeline.process_query(question)
    st.success(answer)

st.sidebar.title("📊 Status")
st.sidebar.success("✅ DB: OK" if os.path.exists(db_path) else "❌ Vazia")
