# streamlit_app.py
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Recipe Search", page_icon="🍲")
st.title("🍲 Recipe Search")

query = st.text_input("Enter a recipe query:", "chicken pasta")
k = st.slider("Top-K results", 1, 20, 5)

if st.button("Search"):
    payload = {"query": query, "k": k, "rerank": False}
    resp = requests.post(f"{API_URL}/search", json=payload, timeout=30)
    if resp.status_code != 200:
        st.error(f"Error {resp.status_code}: {resp.text}")
    else:
        data = resp.json()
        st.write(f"⚡ Took {data['took_ms']} ms, index size={data['index_size']}")
        for hit in data["hits"]:
            st.markdown(f"**Score {hit['score']:.3f}** — {hit['content']}")
            if hit.get("recipe_source"):
                st.caption(f"Source: {hit['recipe_source']}")
            st.divider()
