import anthropic, re, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# FICTIONAL data for learning only. Not real allergen or nutrition information.
DOCS = {
    "allergen_guide.txt": [
        "The Candy Swirl contains milk, soy and peanuts. It is made on shared equipment with tree nuts.",
        "The Classic Burger bun contains wheat and sesame. Classic Burger sauce contains egg and soy.",
        # FIXED DOCUMENT: now explicit about peanuts and tree nuts
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

def tokens(text):
    return re.findall(r"\w+", text.lower())

# ---------- OFFLINE: build BOTH indexes ----------
chunks = [{"source": src, "text": t} for src, texts in DOCS.items() for t in texts]
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)  # dense index
bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])                            # keyword index

MIN_DENSE_SCORE = 0.20   # below this, nothing is relevant enough: don't call the LLM

# ---------- ONLINE: hybrid retrieval with Reciprocal Rank Fusion ----------
def retrieve(question, k=3):
    qv = embedder.encode([question], normalize_embeddings=True)[0]
    dense = vectors @ qv
    sparse = bm25.get_scores(tokens(question))
    rrf = {}
    for ranking in (np.argsort(dense)[::-1], np.argsort(sparse)[::-1]):
        for rank, i in enumerate(ranking):
            rrf[i] = rrf.get(i, 0) + 1 / (60 + rank + 1)     # standard RRF formula
    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    return [(chunks[i], float(dense[i]), float(sparse[i])) for i in top], float(dense.max())

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
SYSTEM = ("You answer customer questions for a burger restaurant using ONLY the numbered sources. "
          "Cite sources like [1]. If the sources do not contain the answer, say you don't know "
          "and suggest asking staff. For allergy questions, end with: "
          "'Please confirm with staff if you have a severe allergy.' Keep answers short.")

def answer(question):
    hits, best_dense = retrieve(question)
    for c, d, s in hits:
        print(f"   [vector {d:.2f} | keyword {s:.2f}] {c['source']}: {c['text'][:60]}...")
    if best_dense < MIN_DENSE_SCORE:
        return "(no LLM call) I don't have information on that. Please ask a staff member."
    context = "\n".join(f"[{i+1}] ({c['source']}) {c['text']}" for i, (c, d, s) in enumerate(hits))
    r = client.messages.create(model=MODEL, max_tokens=500, system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}])
    return "".join(b.text for b in r.content if b.type == "text")

tests = [
    "how do I get a free burger?",
    "can someone with a peanut allergy eat the fries?",
    "does the swirl have nuts?",
    "who is the CEO of the company?",
]
for q in tests:
    print("\nQUESTION:", q)
    print("ANSWER:", answer(q))
