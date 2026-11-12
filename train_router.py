import io, time, contextlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Reuse the SAME router eval cases and the real Haiku supervisor from Day 5
src = open("day5_evals.py").read().split("# ================= SCORECARD")[0]
ns = {}
exec(src, ns)
ROUTER_CASES, supervisor = ns["ROUTER_CASES"], ns["supervisor"]
from rag import embedder   # the same frozen MiniLM embedding model used for RAG

def c(user, prev=None):
    """Turn a message (plus the assistant's previous question, if any) into classifier input."""
    return f"assistant asked: {prev} | customer: {user}" if prev else f"customer: {user}"

def to_text(msgs):
    prev = next((m["content"] for m in reversed(msgs[:-1]) if m["role"] == "assistant"), None)
    return c(msgs[-1]["content"], prev)

# ---------- TRAINING DATA (none of these repeat the Day 5 eval cases: no data leakage) ----------
ORDER = [c(t) for t in [
    "i'd like two classic burgers", "can I get a large fries", "put a medium coke in my cart",
    "remove the fries", "take off the swirl", "what's in my cart?", "how much is my total?",
    "I want to place my order", "let's pay now", "same as last time please",
    "I'll have my regular order", "one swirl please", "add three cokes", "change the coke to large",
    "cancel the burger", "I'm ready to check out", "two burgers and fries",
    "can you add another burger", "and a small coke too"]] + [
    c("large", "What size would you like for your coke?"),
    c("medium please", "Which size of fries would you like?"),
    c("yes, place it", "Should I place your order?"),
    c("just one", "How many burgers would you like?"),
    c("no that's all", "Would you like anything else?")]
INFO = [c(t) for t in [
    "are there peanuts in the swirl", "what allergens are in the burger bun", "is anything gluten free",
    "what time do you open", "are you open late on friday", "how do refunds work",
    "how many calories are in large fries", "what's the healthiest option", "how do loyalty points work",
    "how many points do I need for a free burger", "do the fries have dairy", "is the coke allergen free",
    "what oil do you fry in", "when does the drive-thru close", "can I return my food",
    "which items have sesame", "is there egg in the sauce", "what are your store hours",
    "how many calories is a coke", "do you have low calorie options"]] + [
    c("what about the fries?", "The Candy Swirl contains peanuts.")]
OTHER = [c(t) for t in [
    "hi there", "hello", "thanks!", "who won the game last night", "write me a poem", "what's 2+2",
    "can you help me with my homework", "what's the capital of france", "how are you today",
    "do you like music", "book me a flight", "what's the news", "tell me something funny",
    "translate hello to spanish", "goodbye", "what is AI", "recommend a movie", "how's the traffic"]]

texts = ORDER + INFO + OTHER
labels = ["order"] * len(ORDER) + ["info"] * len(INFO) + ["other"] * len(OTHER)
print(f"Training examples: {len(texts)} (order {len(ORDER)}, info {len(INFO)}, other {len(OTHER)})")

X = embedder.encode(texts, normalize_embeddings=True)   # features: 384 numbers per message
y = np.array(labels)

# ---------- STEP 1: train/test split -> honest accuracy on examples the model never saw ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print("\n=== HELD-OUT TEST (25% of training data, never seen in training) ===")
print(classification_report(y_te, clf.predict(X_te), zero_division=0))

# ---------- STEP 2: after validating, train the final model on ALL examples ----------
clf = LogisticRegression(max_iter=1000).fit(X, y)
n_params = clf.coef_.size + clf.intercept_.size
print(f"Trained parameters: {n_params} (vs billions in an LLM)")

# ---------- STEP 3: the SAME Day 5 router eval: classifier vs Haiku vs cascade ----------
CONFIDENCE = 0.70
print(f"\n=== DAY 5 ROUTER EVAL: classifier vs Haiku (cascade threshold {CONFIDENCE}) ===")
scores = {"clf": 0, "llm": 0, "cascade": 0}
llm_needed, t_clf_total, t_llm_total = 0, 0.0, 0.0
for msgs, expected in ROUTER_CASES:
    t0 = time.perf_counter()
    probs = clf.predict_proba(embedder.encode([to_text(msgs)], normalize_embeddings=True))[0]
    t_clf = time.perf_counter() - t0
    pred, conf = clf.classes_[probs.argmax()], probs.max()

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):       # hide the [supervisor] print
        llm = supervisor({"messages": msgs})["route"]
    t_llm = time.perf_counter() - t0

    cascade = pred if conf >= CONFIDENCE else llm          # LLM only when the classifier is unsure
    llm_needed += conf < CONFIDENCE
    t_clf_total += t_clf
    t_llm_total += t_llm
    scores["clf"] += pred in expected
    scores["llm"] += llm in expected
    scores["cascade"] += cascade in expected
    flag = "" if pred in expected else "   <-- classifier wrong"
    print(f"  clf {pred:<5} ({conf:.2f}, {t_clf*1000:4.0f} ms) | haiku {llm:<5} ({t_llm*1000:4.0f} ms)"
          f"  <- {msgs[-1]['content']}{flag}")

n = len(ROUTER_CASES)
print("\n=========== RESULTS ===========")
print(f"  Trained classifier : {scores['clf']}/{n} correct, avg {t_clf_total/n*1000:.0f} ms, $0, deterministic")
print(f"  Haiku router       : {scores['llm']}/{n} correct, avg {t_llm_total/n*1000:.0f} ms, API cost per message")
print(f"  Cascade            : {scores['cascade']}/{n} correct, LLM calls needed: {llm_needed}/{n}")
