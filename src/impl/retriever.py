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

    # search for 5 chunks using embedding
    def search(self, query: str, top_k: int = 10) -> list[str]:
        search_results = self.datastore.search(query, top_k=5)
        reranked_results = self._rerank(query, search_results, top_k=top_k)
        return reranked_results

    # rank best chunks
    def _rerank(
        self, query: str, search_results: list[str], top_k: int = 10
    ) -> list[str]:
        numbered_docs = "\n\n".join(
            f"[{i}] {doc}" for i, doc in enumerate(search_results)
        )

        response = self.client.chat.completions.create(
            # Model selection: gpt-4o-mini chosen for optimal speed/accuracy/cost ratio 
            # (better latency)
            model="gpt-4o-mini",

            # Messages array follows OpenAI Messages Format specification
            messages=[
                # System message sets JSON contract - GPT must respond EXACTLY as specified
                {"role": "system", "content": "JSON ranker. Return EXACTLY {\"indices\": [0,1,2]} with VALID indices 0 to {len(search_results)-1} ONLY."},
                        
                # User message provides query context + document chunks for ranking task
                # (no hallucinations, 100% parseable)
                {"role": "user", "content": f"""Query: {query}
    Rank these documents 0-{len(search_results)-1} by relevance. Return ONLY:
    {{"indices": [mais_relevante_primeiro, segundo, ...]}}

    Documents:
    {numbered_docs}"""}
            ],
            response_format={"type": "json_object"}
        )
        # Extracts complete JSON response from first (and only) completion choice
        content = response.choices[0].message.content
        
        data = json.loads(content)
    
            # output jus valid indices
        result_indices = data.get("indices", [])
        valid_indices = [i for i in result_indices if isinstance(i, int) and 0 <= i < len(search_results)]

        print(f"✅ Reranked Indices: {valid_indices}(from {len(search_results)} chunks")
        return [search_results[i] for i in valid_indices[:top_k]]