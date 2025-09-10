from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from .retrieval_pipeline import retrievalService
from .models import SearchHit, SearchRequest, SearchResponse

#this is a quick fix
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"


app = FastAPI(title="Recipe Search API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

#default init to none
retriever: retrievalService | None = None

def get_retrieval_service() -> retrievalService:
    global retriever
    if retriever is None:
        retriever = retrievalService()
    return retriever

@app.get("/healthz")
def healthz():
    r = get_retrieval_service()
    return {"status": "ok", "index_size": r.index.ntotal, "model": r.model_name}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    ret = get_retrieval_service()
    start_time = time.perf_counter()
    hits, vec = ret.search(req.query, req.k)
    #RERANK HERE WHEN ITS IMPLEMENTED
    latency = (time.perf_counter() - start_time) * 1000.0
    return SearchResponse(
        query=req.query,
        k=req.k,
        hits=[SearchHit(**h) for h in hits],
        took_ms=round(latency, 2),
        index_size=ret.index.ntotal,
        model=ret.model_name
    )