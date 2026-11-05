import re, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# FICTIONAL data for learning only. Not real allergen or nutrition information.
DOCS = {
    "allergen_guide.txt": [
        "The Candy Swirl contains milk, soy and peanuts. It is made on shared equipment with tree nuts.",
        "The Classic Burger bun contains wheat and sesame. Classic Burger sauce contains egg and soy.",
        "Fries are cooked in a dedicated fryer using a vegetable oil blend. Fries contain no milk, egg, peanuts or tree nuts.",
        "Coke and other fountain drinks contain no major allergens.",
    ],
    "store_policies.txt": [
        "Refunds can be requested within 24 hours of an order by showing the order ID at the counter or in the app.",
        "Loyalty program: customers earn 10 points per dollar spent. 1500 points can be redeemed for a free Classic Burger.",
        "Store hours are 6 AM to 11 PM daily. The drive-thru stays open 24 hours on Fridays and Saturdays.",
    ],
    "nutrition.txt": [
        "Classic Burger has 590 calories. Large fries have 480 calories. Medium Coke has 200 calories.",
        "Lower-calorie choices include small fries at 230 calories and Diet Coke at 0 calories.",
    ],
}

# Fix from last time: common words no longer pollute keyword scores
STOPWORDS = {"the", "a", "an", "is", "of", "to", "do", "does", "i", "how", "what", "who",
             "can", "and", "or", "in", "on", "for", "with", "my", "your", "it", "are",
             "be", "there", "any", "have", "has", "at", "this", "that", "you", "get"}

def tokens(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]

# ---------- OFFLINE: build indexes once ----------
chunks = [{"source": src, "text": t} for src, texts in DOCS.items() for t in texts]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

MIN_DENSE_SCORE = 0.20

# ---------- ONLINE: hybrid search -> re-rank ----------
def search_store_info(question, k=3, candidates=6):
    qv = embedder.encode([question], normalize_embeddings=True)[0]
    dense = vectors @ qv
    if dense.max() < MIN_DENSE_SCORE:
        return {"results": [], "note": "No relevant information in store documents."}

    sparse = bm25.get_scores(tokens(question))
    rrf = {}
    for ranking in (np.argsort(dense)[::-1], np.argsort(sparse)[::-1]):
        for rank, i in enumerate(ranking):
            rrf[i] = rrf.get(i, 0) + 1 / (60 + rank + 1)
    cand = sorted(rrf, key=rrf.get, reverse=True)[:candidates]

    scores = reranker.predict([(question, chunks[i]["text"]) for i in cand])
    order = np.argsort(scores)[::-1][:k]
    return {"results": [{"source": chunks[cand[j]]["source"],
                         "text": chunks[cand[j]]["text"],
                         "rerank_score": round(float(scores[j]), 2)} for j in order]}

# Test the service on its own (runs only with: python3 rag.py)
if __name__ == "__main__":
    for q in ["how do I get a free burger?", "who is the CEO of the company?"]:
        print("\nQ:", q)
        for r in search_store_info(q).get("results", []) or ["(no results)"]:
            print("  ", r)
