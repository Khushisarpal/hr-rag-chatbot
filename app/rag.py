"""Pipeline RAG : decoupage, embeddings, base vectorielle (LangChain), generation."""
import os
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import HashingVectorizer

STOP = set("""le la les un une des de du d l et ou a au aux en dans pour par sur avec sans est sont
ce cet cette ces qui que quoi quel quelle quels quelles il elle on je tu nous vous ils elles mon ma mes
ton ta tes son sa ses notre votre leur y ne pas plus se si comment combien faut faire puis peut ai as
avons ont etre avoir tout tous""".split())

MIN_SCORE = float(os.getenv("MIN_SCORE", "0.15"))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def analyzer(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", _norm(text))
    return [t[:6] for t in toks if t not in STOP and len(t) > 1]  # racinisation grossiere


class HashEmbeddings(Embeddings):
    """Embeddings lexicaux sans modele a telecharger (deploiement leger, tests deterministes)."""

    def __init__(self, n_features: int = 4096):
        self.vec = HashingVectorizer(n_features=n_features, analyzer=analyzer,
                                     alternate_sign=False, norm="l2")

    def embed_documents(self, texts):
        return self.vec.transform(texts).toarray().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def load_chunks(folder: Path) -> list[Document]:
    headers = [("#", "theme"), ("##", "section")]
    md = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    rc = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks: list[Document] = []
    for path in sorted(folder.glob("*.md")):
        for d in md.split_text(path.read_text(encoding="utf-8")):
            d.metadata["source"] = path.name
            chunks.extend(rc.split_documents([d]))
    return chunks


class HRBot:
    def __init__(self, folder: Path):
        self.store = InMemoryVectorStore(HashEmbeddings())
        self.store.add_documents(load_chunks(folder))

    def retrieve(self, question: str, k: int = 3):
        return self.store.similarity_search_with_score(question, k=k)

    def answer(self, question: str, k: int = 3) -> dict:
        hits = [(d, s) for d, s in self.retrieve(question, k) if s >= MIN_SCORE]
        if not hits:
            return {"answer": "Je n'ai pas trouvé cette information dans les documents RH. "
                              "Contactez l'équipe RH.", "sources": [], "grounded": False}
        context = "\n\n".join(d.page_content for d, _ in hits)
        sources = [{"file": d.metadata["source"], "section": d.metadata.get("section", ""),
                    "score": round(float(s), 3)} for d, s in hits]
        return {"answer": self._generate(question, context, hits), "sources": sources, "grounded": True}

    def _generate(self, question, context, hits) -> str:
        if os.getenv("ANTHROPIC_API_KEY"):  # mode LLM optionnel
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model=os.getenv("LLM_MODEL", "claude-haiku-5-5"), max_tokens=400)
                prompt = ("Réponds en français, uniquement à partir du contexte. Si l'information "
                          f"manque, dis-le.\n\nContexte:\n{context}\n\nQuestion: {question}")
                return llm.invoke(prompt).content
            except Exception:
                pass
        # Mode extractif : phrases les plus proches de la question
        q = set(analyzer(question))
        sents = [s.strip() for d, _ in hits for s in re.split(r"(?<=[.!?])\s+", d.page_content) if s.strip()]
        ranked = sorted(sents, key=lambda s: len(q & set(analyzer(s))), reverse=True)
        best = [s for s in ranked[:2] if q & set(analyzer(s))]
        return " ".join(best) or hits[0][0].page_content
