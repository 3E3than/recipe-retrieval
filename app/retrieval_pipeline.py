import os, json, glob, re
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "index" / "faiss"
META_PATH = ROOT / "index" / "meta.json"

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index" / "faiss"
META_PATH = ROOT / "index" / "meta.json"
EMBED_MODEL = "intfloat/e5-small-v2"

#MUST BE SAME AS INGESTION

def prepare_query(q: str) -> str:
    q = q.strip()
    return f"query: {q}"

#exact same process as ingestion pipeline
@torch.inference_mode()
def encode(queries: List[str], tokenizer, model, device) -> np.ndarray:
    batch = [prepare_query(q) for q in queries]
    encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    model_output = model(**encoded).last_hidden_state
    attention = encoded["attention_mask"].unsqueeze(-1)    
    pooled = (model_output * attention).sum(1) / attention.sum(1).clamp(min=1e-9)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    result = pooled.cpu().numpy().astype("float32")
    return result 

class retrievalService:
    def __init__(self):
        # Load meta (records contains original mapping)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.model_name: str = meta.get("model", EMBED_MODEL)
        self.records: List[Dict[str, str]] = meta["records"]   # [{text, source, ...}, ...]

        # Load FAISS index
        self.index = faiss.read_index(str(INDEX_DIR / "e5.index"))
        self.dim = self.index.d

        # Load embedder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def search(self, query: str, k: int = 10):
        results = []
        #first encode query
        query_vector = encode([query], self.tok, self.model, self.device)
        #search
        scores, ids = self.index.search(query_vector, k)
        # get text
        for rank, (id, score) in enumerate(zip(ids[0], scores[0])):
            if id < 0:
                continue
            record_metadata = self.records[id]
            results.append({
                "id": id,
                "score": score,
                "content": record_metadata.get("content", ""),
                "recipe_source": record_metadata.get("recipe_name", "")                
            })
        return (results, query_vector)
    
    ## FROM GPT: IMPROVEMENTS ##
    # Cross-encoder reranker (placeholder)
    # You can later plug sentence-transformers cross-encoder here if you want:
    # from sentence_transformers import CrossEncoder
    # self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # and sort topN by rerank score.
    def rerank_placeholder(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # No-op for now; stable interface for future upgrade.
        return hits
