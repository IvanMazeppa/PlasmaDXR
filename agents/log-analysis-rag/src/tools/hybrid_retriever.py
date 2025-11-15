"""
Hybrid Retrieval System (BM25 + FAISS)
Based on NVIDIA BAT.AI multi-agent RAG architecture
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document


class PlasmaDXHybridRetriever:
    """
    Hybrid retrieval combining BM25 (keyword) and FAISS (semantic)
    Optimized for PlasmaDX log analysis and GPU debugging
    """

    def __init__(
        self,
        log_dirs: List[str],
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        chroma_db_path: Optional[str] = None,
        top_k: int = 20
    ):
        self.log_dirs = [Path(d) for d in log_dirs]
        self.embedding_model = embedding_model
        self.chroma_db_path = chroma_db_path
        self.top_k = top_k

        # Will be initialized on first retrieval
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.faiss_retriever: Optional[FAISS] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

        # Document corpus
        self.documents: List[Document] = []

    def load_documents(self) -> List[Document]:
        """
        Load all log files, PIX captures, and buffer dumps
        Returns list of LangChain Documents
        """
        documents = []

        for log_dir in self.log_dirs:
            if not log_dir.exists():
                print(f"⚠️  Log directory not found: {log_dir}")
                continue

            # Load text files (logs)
            for log_file in log_dir.glob("**/*.txt"):
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Split into line-based chunks for better granularity
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip():  # Skip empty lines
                            doc = Document(
                                page_content=line,
                                metadata={
                                    "source": str(log_file),
                                    "line": i + 1,
                                    "type": "log"
                                }
                            )
                            documents.append(doc)

                except Exception as e:
                    print(f"⚠️  Failed to load {log_file}: {e}")

            # TODO: Load PIX buffer dumps (binary files require special parsing)
            # TODO: Load PIX capture metadata

        print(f"✅ Loaded {len(documents)} document chunks from {len(self.log_dirs)} directories")
        self.documents = documents
        return documents

    def initialize_retrievers(self):
        """Initialize BM25 and FAISS retrievers"""
        if not self.documents:
            self.load_documents()

        if len(self.documents) == 0:
            raise ValueError("No documents loaded - check log directories exist and contain .txt files")

        # 1. BM25 Retriever (keyword-based, exact matching)
        # Critical for DXR error codes: "DXGI_ERROR_DEVICE_REMOVED", "D3D12_ERROR_*"
        print("🔧 Initializing BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=self.top_k
        )

        # 2. FAISS Retriever (semantic similarity)
        # Matches conceptual issues: "temporal instability" → reservoir problems
        print("🔧 Initializing FAISS retriever with NVIDIA embeddings...")
        embeddings = NVIDIAEmbeddings(model=self.embedding_model)

        self.faiss_retriever = FAISS.from_documents(
            self.documents,
            embeddings
        ).as_retriever(search_kwargs={"k": self.top_k})

        # 3. Ensemble Retriever (50/50 weighting)
        print("🔧 Creating ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.faiss_retriever],
            weights=[0.5, 0.5]  # Equal weight for keyword + semantic
        )

        print("✅ Hybrid retrieval system initialized")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query
        Uses hybrid BM25 + FAISS approach
        """
        if not self.ensemble_retriever:
            self.initialize_retrievers()

        print(f"🔍 Retrieving documents for: '{query}'")
        docs = self.ensemble_retriever.get_relevant_documents(query)

        print(f"✅ Retrieved {len(docs)} documents")
        return docs


# Example usage
if __name__ == "__main__":
    # Test with PlasmaDX log directories
    retriever = PlasmaDXHybridRetriever(
        log_dirs=[
            "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/logs",
            "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/PIX/buffer_dumps",
        ]
    )

    # Test query
    test_query = "Why is RTXDI M5 showing patchwork pattern?"
    docs = retriever.retrieve(test_query)

    print("\n" + "="*60)
    print("Top 5 Results:")
    print("="*60)
    for i, doc in enumerate(docs[:5]):
        print(f"\n{i+1}. {doc.metadata['source']}:{doc.metadata.get('line', '?')}")
        print(f"   {doc.page_content[:100]}...")
