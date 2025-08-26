import os, json, glob, re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "index" / "faiss"
META_PATH = ROOT / "index" / "meta.json"

EMBED_MODEL = "intfloat/e5-small-v2"
WINDOW_STEPS = 2     # number of consecutive steps per chunk
OVERLAP_STEPS = 1    # how many steps to overlap between chunks

def normalize(s: str) -> str:
    # collapse whitespace, fix missing spaces after punctuation, strip
    s = re.sub(r"\s+", " ", s)
    # re.sub does (pattern, sub, target)
    s = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", s)
    return s.strip()

def get_title(recipe_id: str) -> str:
    # delicious-coconut-truffles -> Delicious Coconut Truffles
    return re.sub(r"\s+", " ", recipe_id.replace("-", " ")).title()

def raw_to_steps(js_file: dict) -> List[Tuple[int, str]]:
    # turn json files into step number and instruction pairs
    result = []
    raw_steps = js_file.get("steps", [])
    for step in raw_steps:
        title  = normalize(step.get("title", ""))
        body = normalize(step.get("body", ))
        #skip intro parts and don't add title if it doesn't have useful titles
        if title.lower().startswith("introduction"):
            continue
        if title and not title.lower().startswith("step"):
            fulltext = f"{title}. {body}"
        else:
            fulltext = body if body else title
        finaltext = normalize(fulltext)
        if finaltext:
            result.append((int(step.get("id", len(result))), finaltext))
    finalresult = [(i + 1, t) for i, (_, t)in enumerate(sorted(result, key=lambda x: x[0]))]
    return finalresult

def chunk_steps(steps: List[Tuple[int, str]]):
    """turns steps into chunked version with start, end, and the text"""
    if not steps:
        return []
    chunks = []
    i = 0
    while i < len(steps):
        group = steps[i:i+WINDOW_STEPS]
        nums, texts = [s for s, _ in group], [t for _, t in group]
        if not texts:
            break
        start, end = nums[0], nums[-1]
        combined = "".join(texts)
        chunks.append((start, end, normalize(combined)))
        if i + WINDOW_STEPS >= len(steps):
            break
        i += (WINDOW_STEPS - OVERLAP_STEPS) if WINDOW_STEPS > OVERLAP_STEPS else 1
    return chunks

def load_json()-> List[Dict]:
    result = []
    for file in sorted(glob.glob(DATA_DIR / "*.json")):
        try:
            with open(file, encoding="utf-8", errors="ignore") as jsonfile:
                object = json.load(jsonfile)
        except Exception as e:
            print(f"error with json file {file}: {e}")
            continue
        raw_recipe = object.get("recipe_id") or Path(file).stem
        recipe_name = get_title(raw_recipe)
        raw_steps = raw_to_steps(object)
        chunked_steps = chunk_steps(raw_steps)
        for start, end, content in chunked_steps:
            result.append({
                "recipe_id": raw_recipe,
                "recipe_name": recipe_name,
                "start_step": start,
                "end_step": end,
                "content": content
            })
    return result

def encode(load: List[str], tokenizer, model, device):
    embeddings, batch_size = [], 32
    for i in range(0, len(load), batch_size):
        #e5 embeddings require passages to be labeled passage
        batch = [f"passage: {p}" for p in load[i:i+batch_size]]
        #step 1: encode each passage in batch, here we are truncating down to 512, and padding up to 512, and returning as pytorch tensors
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        #step 2: run tokenized tensors through model; 
        model_output = model(**encoded).last_hidden_state
        #step 3: get the attention_mask, and add 1 dimension so it works with model_output
        attention = encoded["attention_mask"].unsqueeze(-1)
        #creating one vector per passage, then normalize
        #here, we turn each passage into one single vector, and make sure we are ignoring the attention mask 0s
        pooled = (model_output * attention).sum(1) / attention.sum(1).clamp(min=1e-9)
        # then, scale each vector/passage down to standardize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        
