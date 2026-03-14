from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import time
import threading
from typing import Dict, List, Optional
from openai import RateLimitError
from interface import (
    BaseDatastore,
    BaseIndexer,
    BaseRetriever,
    BaseResponseGenerator,
    BaseEvaluator,
    EvaluationResult,
)


class _RateLimiter:
    """Token bucket rate limiter — garante no máximo `max_rpm` chamadas por minuto."""

    def __init__(self, max_rpm: int = 10):
        self._max_rpm = max_rpm
        self._interval = 60.0 / max_rpm  # segundos entre cada chamada
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


def _with_retry(fn, rate_limiter: _RateLimiter, max_retries: int = 4):
    """Executa fn respeitando o rate limiter e faz retry com backoff exponencial."""
    for attempt in range(max_retries):
        try:
            rate_limiter.acquire()
            return fn()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 15 * (2 ** attempt)  # 15s → 30s → 60s → 120s
            print(f"⚠️  Rate limit (tentativa {attempt + 1}/{max_retries}). Aguardando {wait}s...")
            time.sleep(wait)


@dataclass
class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""

    datastore: BaseDatastore
    indexer: BaseIndexer
    retriever: BaseRetriever
    response_generator: BaseResponseGenerator
    evaluator: Optional[BaseEvaluator] = None
    max_workers: int = 2        # Seguro para free tier (15 RPM)
    max_rpm: int = 10           # Margem de segurança abaixo dos 15 RPM do Gemini
    _rate_limiter: _RateLimiter = field(init=False)

    def __post_init__(self):
        self._rate_limiter = _RateLimiter(max_rpm=self.max_rpm)

    def reset(self) -> None:
        self.datastore.reset()

    def add_documents(self, documents: List[str]) -> None:
        items = self.indexer.index(documents)
        self.datastore.add_items(items)
        print(f"✅ Added {len(items)} items to the datastore.")

    def process_query(self, query: str) -> str:
        search_results = self.retriever.search(query)
        print(f"✅ Found {len(search_results)} results for query: {query}\n")

        for i, result in enumerate(search_results):
            print(f"🔍 Result {i + 1}: {result}\n")

        response = _with_retry(
            fn=lambda: self.response_generator.generate_response(query, search_results),
            rate_limiter=self._rate_limiter,
        )
        return response

    def evaluate(self, sample_questions: List[Dict[str, str]]) -> List[EvaluationResult]:
        questions = [item["question"] for item in sample_questions]
        expected_answers = [item["answer"] for item in sample_questions]
        pairs = list(zip(questions, expected_answers))

        results: List[Optional[EvaluationResult]] = [None] * len(pairs)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._evaluate_single_question, q, a): i
                for i, (q, a) in enumerate(pairs)
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"❌ Erro na questão {i + 1}: {e}")
                    results[i] = EvaluationResult(
                        question=pairs[i][0],
                        response="ERROR",
                        expected_answer=pairs[i][1],
                        is_correct=False,
                        reasoning=str(e),
                    )

        for i, result in enumerate(results):
            emoji = "✅" if result.is_correct else "❌"
            print(f"{emoji} Q {i + 1}: {result.question}\n")
            print(f"Response: {result.response}\n")
            print(f"Expected Answer: {result.expected_answer}\n")
            print(f"Reasoning: {result.reasoning}\n")
            print("--------------------------------")

        number_correct = sum(r.is_correct for r in results)
        print(f"✨ Total Score: {number_correct}/{len(results)}")
        return results

    def _evaluate_single_question(self, question: str, expected_answer: str) -> EvaluationResult:
        response = self.process_query(question)
        return _with_retry(
            fn=lambda: self.evaluator.evaluate(question, response, expected_answer),
            rate_limiter=self._rate_limiter,
        )
