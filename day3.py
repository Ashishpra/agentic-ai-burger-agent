import anthropic, numpy as np
from sentence_transformers import SentenceTransformer

# FICTIONAL data for learning only. Not real allergen or nutrition information.
# In production these come from PDFs, SharePoint, S3, Confluence, etc.
DOCS = {
    "allergen_guide.txt": [
        "The Candy Swirl contains milk, soy and peanuts. It is made on shared equipment with tree nuts.",
        "The Classic Burger bun contains wheat and sesame. Classic Burger sauce contains egg and soy.",
        "Fries are cooked in a dedicated fryer using a vegetable oil blend. Fries contain no milk, egg or nuts.",
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

# ---------- OFFLINE: chunk + embed (here each sentence is already one chunk) ----------
chunks = [{"source": src, "text": t} for src, texts in DOCS.items() for t in texts]
print("Loading embedding model (first run downloads it)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
print(f"Indexed {len(chunks)} chunks. Each embedding has {vectors.shape[1]} numbers.")

# ---------- ONLINE: retrieve ----------
def retrieve(question, k=3):
    qv = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ qv                      # cosine similarity: closer meaning = higher
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]

# ---------- ONLINE: generate with citations ----------
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
SYSTEM = ("You answer customer questions for a burger restaurant using ONLY the numbered sources. "
          "Cite sources like [1]. If the sources do not contain the answer, say you don't know "
          "and suggest asking staff. For allergy questions, end with: "
          "'Please confirm with staff if you have a severe allergy.' Keep answers short.")

def answer(question):
    hits = retrieve(question)
    for c, s in hits:
        print(f"   [retrieved {s:.2f}] {c['source']}: {c['text'][:70]}...")
    context = "\n".join(f"[{i+1}] ({c['source']}) {c['text']}" for i, (c, s) in enumerate(hits))
    r = client.messages.create(model=MODEL, max_tokens=500, system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}])
    return "".join(b.text for b in r.content if b.type == "text")

tests = [
    "does the swirl have nuts?",
    "can someone with a peanut allergy eat the fries?",
    "how do I get a free burger?",
    "when does the drive thru close on saturday?",
    "is the classic burger gluten free?",
    "who is the CEO of the company?",
]
for q in tests:
    print("\nQUESTION:", q)
    print("ANSWER:", answer(q))
