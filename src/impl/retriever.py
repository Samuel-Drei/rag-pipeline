from interface.base_datastore import BaseDatastore
from interface.base_retriever import BaseRetriever
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import streamlit as st

load_dotenv()

def _get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")
   

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore
        self.client = OpenAI(api_key=_get_api_key())

    def search(self, query: str, top_k: int = 10) -> list[str]:
        search_results = self.datastore.search(query, top_k=top_k * 3)
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    def _rerank(
        self, query: str, search_results: list[str], top_k: int = 10
    ) -> list[str]:
        # Monta o prompt pedindo ao GPT para ordenar por relevância
        numbered_docs = "\n\n".join(
            f"[{i}] {doc}" for i, doc in enumerate(search_results)
        )
        prompt = f"""Given the query below, rank the documents by relevance.
Return ONLY a JSON array with the indices of the top {top_k} most relevant documents, ordered from most to least relevant.
Example: [2, 0, 5, 1, 3]

Query: {query}

Documents:
{numbered_docs}"""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.choices[0].message.content.strip()
        # Extrai o JSON da resposta
        start = raw.find("[")
        end = raw.rfind("]") + 1
        result_indices = json.loads(raw[start:end])[:top_k]

        print(f"✅ Reranked Indices: {result_indices}")
        return [search_results[i] for i in result_indices]
