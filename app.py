import streamlit as st
import os
import sys
import glob
from main import create_pipeline, get_files_in_directory

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="RAG Pipeline", page_icon="🔍")
st.title("🔍 RAG Pipeline v2.0")
st.info("88% accuracy | 5.2s avg response")  # Seu benchmark!

# SEM CACHE no pipeline (evita freeze)
@st.cache_data(ttl=300)  # Cache só queries (5min)
def check_db_status():
    db_path = "data/sample-lancedb/rag-table.lance"
    return os.path.exists(db_path)

# AUTO-REINDEX SEMPRE QUE DB VAZIA
if not check_db_status():
    with st.spinner("🔄 Indexing PDFs (1st time only)..."):
        pipeline = create_pipeline()
        docs = get_files_in_directory("sample_data/source/")
        pipeline.reset()  # Limpa se corrompido
        pipeline.add_documents(docs)
        st.success(f"✅ Indexed {len(docs)} documents!")
else:
    pipeline = create_pipeline()

question = st.text_input("👇 Ask about the PDFs:", key="question")
if st.button("🔍 Search RAG", type="primary") and question:
    with st.spinner("🤖 RAG processing..."):
        answer = pipeline.process_query(question)
    
    # VERDE VOLTA!
    st.success("✅ **Answer:**")
    st.markdown(answer)

# Status DB
if st.sidebar.checkbox("DB Status"):
    st.sidebar.metric("Documents Indexed", "Ready" if check_db_status() else "Empty")
