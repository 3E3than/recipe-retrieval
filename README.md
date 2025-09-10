# 🍳 Recipe Search

**Recipe Search API** is a FastAPI-based service that lets you search through cooking recipes using semantic retrieval.  
It supports **procedural queries** (e.g., *"How long to cook steak?"*, *"When should I flip pancakes?"*) by indexing step-by-step instructions and returning the most relevant steps.

---

## ✨ Features
- **Ingest JSON recipes** (with `recipe_id` + `steps[]`) or plain text/Markdown recipes.
- **Embeddings with E5** (`intfloat/e5-small-v2`) → turns recipe steps into semantic vectors.
- **FAISS index** for fast similarity search (Flat or compressed IVFPQ).
- (IN PROGRESS)**Retriever + Reranker** pipeline for high-precision results.
- (IN PROGRESS)**FastAPI service** with:
  - `POST /search` → semantic search for procedural questions
  - `GET /healthz` → health check endpoint

---

## 🗂 Project Structure

```text
recipe-search-api/
│
├── app/
│ ├── ingest_json.py # Build FAISS index from JSON or text recipes
│ ├── pipeline.py # Retriever + Reranker logic
│ ├── models.py # Pydantic schemas for FastAPI
│ ├── service.py # FastAPI app (search + healthz endpoints)
│ └── smoke_search.py # Local CLI search tester
│
├── data/
│ └── raw/ # Recipe JSON files go here
│
├── index/
│ └── faiss/ # Generated FAISS index (e5.index / e5_ivfpq.index)
│ └── meta.json # Chunk metadata (maps index to source)
│
├── tests/ # Basic tests for ingestion + API
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Setup & Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/3E3than/recipe-search-api.git
   cd recipe-search-api
   ```

2. **Install dependencies**
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
3. **Add recipe data**
    Unzip the file and place all JSON recipe files into data/raw/
    ```bash
    Each JSON should contain:
    {
    "recipe_id": "delicious-coconut-truffles",
    "steps": [
        { "id": 1, "title": "Step 1: Ingredients", "body": "..." },
        { "id": 2, "title": "Step 2: Mix", "body": "..." }
    ]
    }
    ```
4. **Build the FAISS index**
    ```bash
        python -m app.ingest_json
    ```
---

## 🛠 How Does This Work?

This project is built on **semantic search**: essentially, we turn recipe steps into vector embeddings, store them in a FAISS index, and retrieve the most relevant ones for a user’s query based on the vector formed by the user's questions and the recipe vectors. Here's the steps that I've done:

### 1. Embedding
- Each recipe is split into small **chunks** (usually 2–3 consecutive steps).  
- We use the [E5 model](https://huggingface.co/intfloat/e5-small-v2) to encode text into a **vector** (a list of 384 numbers).  
- Similar meaning essentially equates to similar vectors.  
  - Example:  
    - `"How long to cook steak?"` → `[0.12, -0.45, 0.83, ...]`  
    - `"Grill steak 4 minutes each side."` → `[0.10, -0.48, 0.80, ...]`  
    These two vectors will be close together in “vector space.”

### 2. Indexing
- All vectors are stored in a **FAISS index** (Facebook AI Similarity Search).  
- FAISS is optimized for finding “nearest neighbors” in high-dimensional vector space quickly.  
- We also save **metadata** (recipe ID, step numbers, original text) so results can be tied back to the source recipe.

### 3. Retrieval
1. A user sends a query, e.g. `"When should I flip pancakes?"`.  
2. The query is embedded with the same model and the same steps(`query: …` format).  
3. FAISS finds the closest matching recipe-step vectors.  
4. The API returns those chunks along with their metadata:
