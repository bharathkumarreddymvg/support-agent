import argparse
import logging
import os
import random
import re
from typing import List, Optional, Tuple

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
# --------------- Logging Setup ---------------
LOG_FILE = "support_bot_log.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# --------------- Utilities ---------------

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is not installed. Install it or provide a TXT document.")
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            pages.append(text)
        except Exception as e:
            logging.warning(f"Failed to extract text from page {i}: {e}")
    return "\n\n".join(pages)
def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str) -> List[str]:
    """
    Split text into logical sections.
    Strategy:
      - Prefer double-newline paragraph splits.
      - If too few sections, fallback to sentence-level grouping (~4 sentences per section).
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) >= 5:
        return paras
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sections, buf = [], []
    for s in sentences:
        buf.append(s)
        if len(buf) >= 4:
            sections.append(" ".join(buf).strip())
            buf = []
    if buf:
        sections.append(" ".join(buf).strip())
    return sections
def keyword_score(query: str, section: str) -> int:
    q_tokens = set(re.findall(r"\w+", query.lower()))
    s_tokens = set(re.findall(r"\w+", section.lower()))
    return len(q_tokens & s_tokens)


# --------------- Agent ---------------
class SupportBotAgent:
    def __init__(
        self,
        document_path: str,
        qa_model_name: str = "distilbert-base-uncased-distilled-squad",
        embedder_name: str = "all-MiniLM-L6-v2",
        seed: Optional[int] = 42,
    ):
        if seed is not None:
            random.seed(seed)

        self.document_path = document_path
        self.document_text = self._load_document(document_path)
        self.document_text = clean_text(self.document_text)
        self.sections = split_into_sections(self.document_text)

        logging.info(f"Loaded document: {document_path}")
        logging.info(f"Total sections: {len(self.sections)}")

        # Initialize models
        self.qa_model = pipeline("question-answering", model=qa_model_name)
        self.embedder = SentenceTransformer(embedder_name)

        # Pre-compute embeddings
        self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)
        logging.info("Computed section embeddings.")
    def _load_document(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in [".txt", ".md"]:
            text = read_txt(path)
        elif ext == ".pdf":
            text = read_pdf(path)
        else:
            raise ValueError(f"Unsupported document type: {ext}. Use .txt/.md/.pdf")
        return text

    def find_relevant_section(self, query: str) -> Tuple[str, int]:
        """
        Retrieve the most relevant section with a hybrid strategy:
          1) Semantic similarity (cosine sim)
          2) Keyword overlap as tie-breaker/backup
        Returns (section_text, section_index)
        """
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(query_embedding, self.section_embeddings)[0]
        best_idx = int(sims.argmax())

        best_section = self.sections[best_idx]
        best_kw = keyword_score(query, best_section)
        neighborhood = sorted(range(len(self.sections)), key=lambda i: float(sims[i]), reverse=True)[:3]
        for i in neighborhood:
            kw = keyword_score(query, self.sections[i])
            if kw > best_kw:
                best_idx = i
                best_section = self.sections[i]
                best_kw = kw

        logging.info(f"Found relevant section index={best_idx} for query='{query[:80]}'")
        return best_section, best_idx
    def answer_query(self, query: str) -> str:
        """
        Use QA pipeline with best section as context.
        Fallbacks:
          - If context is empty or QA score is low, provide a graceful message.
        """
        context, idx = self.find_relevant_section(query)
        if not context or len(context) < 30:
            logging.info("Context too short; triggering fallback.")
            return "I don't have enough information to answer that from the provided document."

        try:
            result = self.qa_model(question=query, context=context)
            answer = result.get("answer", "").strip()
            score = float(result.get("score", 0.0))
        except Exception as e:
            logging.error(f"QA pipeline error: {e}")
            return "I ran into an issue while generating an answer. Please try rephrasing your question."

        logging.info(f"QA score={score:.4f} section_idx={idx}")
        if not answer or score < 0.12:
            return "I don't have enough information to answer that from the provided document."
        return answer

    def get_feedback(self, response: str) -> str:
        feedback = random.choice(["good", "too vague", "not helpful"])
        logging.info(f"Feedback received: {feedback}")
        return feedback

    def adjust_response(self, query: str, response: str, feedback: str) -> str:
        if feedback == "too vague":
            context, _ = self.find_relevant_section(query)
            excerpt = context[:160].replace("\n", " ").strip()
            updated = f"{response} More detail: {excerpt} ..."
            logging.info("Adjusted response for 'too vague' by adding context excerpt.")
            return updated
        elif feedback == "not helpful":
            refined_query = query.strip() + " Please be specific and include steps."
            new_response = self.answer_query(refined_query)
            logging.info("Adjusted response for 'not helpful' by refining the query.")
            return new_response
        else:
            return response

    def run(self, queries: List[str], max_iters: int = 2) -> None:
        for q in queries:
            logging.info(f"Processing query: {q}")
            initial = self.answer_query(q)
            print(f"Initial Response to '{q}': {initial}")
            response = initial

            for _ in range(max_iters):
                feedback = self.get_feedback(response)
                print(f"Feedback: {feedback}")
                if feedback == "good":
                    break
                response = self.adjust_response(q, response, feedback)
                print(f"Updated Response to '{q}': {response}")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document-trained customer support bot with retrieval, QA, and feedback."
    )
    parser.add_argument(
        "--document",
        type=str,
        default="faq.txt",
        help="Path to the document (.txt/.md/.pdf)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="*",
        default=[
            "How do I reset my password?",
            "What's the refund policy?",
            "How do I contact support?",
            "How do I fly to the moon?",  # Out-of-scope
        ],
        help="Queries to process (space-separated).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for feedback simulation (set None for fully random).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    agent = SupportBotAgent(document_path=args.document, seed=args.seed)
    agent.run(args.queries)


if __name__ == "__main__":
    main()
