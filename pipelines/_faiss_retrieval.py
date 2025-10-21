from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from chainlite import chain
from pipelines.dialogue_state import DialogueState

class FAISSRetriever:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.index = None
        self.model = SentenceTransformer(model_name)
        self.load_index(index_path)

    def load_index(self, index_path: str):
        """Load the FAISS index from disk."""
        import faiss
        self.index = faiss.read_index(index_path)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents."""
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu().numpy()
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for i in range(k):
            results.append({
                "score": float(D[0][i]),
                "index": I[0][i],
                "content": f"Document {I[0][i]}: Score {D[0][i]:.2f}"
            })
        return results

@chain
async def faiss_retrieval_stage(state: DialogueState):
    """
    Retrieves relevant text segments from FAISS index.
    """
    retriever = FAISSRetriever(
        index_path=state.config.faiss_index_path,
        model_name=state.config.faiss_model_name
    )
    
    # Get query from current turn
    query = state.current_turn.user_utterance
    
    # Retrieve top-k results
    faiss_results = retriever.retrieve(query, k=state.config.faiss_k)
    
    # Store results in dialogue state
    state.current_turn.faiss_results = faiss_results
