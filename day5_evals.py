from rag import search_store_info

# Load the real nodes from day4.py without starting its chat loop
src = open("day4.py").read().split("# ---------- App loop")[0]
ns = {}
exec(src, ns)
supervisor, info_agent, client = ns["supervisor"], ns["info_agent"], ns["client"]
JUDGE_MODEL = "claude-haiku-5-5"

def u(t): return {"role": "user", "content": t}
def a(t): return {"role": "assistant", "content": t}

# ================= A. RETRIEVAL EVAL (free: no LLM) =================
RETRIEVAL_CASES = [   # (question, text expected in the right chunk; None = expect no results)
    ("does the swirl have nuts?", "peanuts"),
    ("how do I get a free burger?", "Loyalty program"),
    ("can someone with a peanut allergy eat the fries?", "Fries are cooked"),
    ("when does the drive thru close on saturday?", "drive-thru"),
    ("is the classic burger gluten free?", "wheat"),
    ("anything with sesame in the burger bun?", "sesame"),
    ("how long do I have to ask for my money back?", "Refunds"),
    ("how many calories in a classic burger?", "590 calories"),
    ("who is the CEO of the company?", None),
    ("what's the wifi password?", None),
]

def eval_retrieval():
    print("\n=== A. RETRIEVAL ===")
    hit1 = hit3 = 0
    for q, expect in RETRIEVAL_CASES:
        texts = [r["text"] for r in search_store_info(q).get("results", [])]
        if expect is None:
            ok1 = ok3 = len(texts) == 0
        else:
            ok1 = bool(texts) and expect in texts[0]
            ok3 = any(expect in t for t in texts)
        hit1 += ok1
        hit3 += ok3
        print(f"  {'PASS' if ok1 else ('TOP3' if ok3 else 'FAIL')}  {q}")
    n = len(RETRIEVAL_CASES)
    return f"hit@1 {hit1}/{n}, hit@3 {hit3}/{n}"

# ================= B. ROUTER EVAL (repeated runs) =================
ROUTER_CASES = [   # (conversation, set of acceptable routes)
    ([u("add a classic burger")], {"order"}),
    ([u("does the swirl have nuts?")], {"info"}),
    ([u("tell me a joke")], {"other"}),
    ([u("add some fries"), a("Which size of fries would you like?"), u("small")], {"order"}),
    ([u("gimme the usual")], {"order"}),
    ([u("how many calories in a classic burger?")], {"info"}),
    ([u("can I get a refund for my last order?")], {"info"}),
    ([u("checkout please")], {"order"}),
    ([u("add a swirl, does it have nuts?")], {"order", "info"}),   # mixed intent
    ([u("what's the weather today?")], {"other"}),
]
RUNS = 3

def eval_router():
    print(f"\n=== B. ROUTER ({RUNS} runs each) ===")
    correct = consistent = 0
    for msgs, expected in ROUTER_CASES:
        routes = [supervisor({"messages": msgs})["route"] for _ in range(RUNS)]
        ok = all(r in expected for r in routes)
        same = len(set(routes)) == 1
        correct += ok
        consistent += same
        print(f"  {'PASS' if ok else 'FAIL'}  {'stable  ' if same else 'UNSTABLE'}  "
              f"{routes}  <- {msgs[-1]['content']}")
    n = len(ROUTER_CASES)
    return f"accuracy {correct}/{n}, consistent {consistent}/{n}"

# ================= C. INFO AGENT BEHAVIOR (code graders + LLM judge) =================
INFO_CASES = [
    {"q": "does the swirl have nuts?", "allergy": True, "answerable": True},
    {"q": "can someone with a peanut allergy eat the fries?", "allergy": True, "answerable": True},
    {"q": "how do I get a free burger?", "allergy": False, "answerable": True},
    {"q": "who is the CEO of the company?", "allergy": False, "answerable": False},
]
UNKNOWN_PHRASES = ["don't know", "do not know", "no information", "couldn't find", "not able to find"]

def judge_faithful(question, answer):
    hits = search_store_info(question).get("results", [])
    sources = "\n".join(f"[{r['source']}] {r['text']}" for r in hits)      # FIX 1: judge sees file names
    prompt = (f"Sources:\n{sources}\n\nAnswer:\n{answer}\n\n"
              "Is every factual claim in the Answer supported by the Sources? Citing the source "
              "file names above is allowed. Ignore safety advice like 'confirm with staff'. "
              "Reply with PASS or FAIL first, then one short reason.")
    for attempt in range(2):                                               # FIX 2: retry empty output
        r = client.messages.create(model=JUDGE_MODEL, max_tokens=1000,      # FIX 3: room to think
                                   messages=[{"role": "user", "content": prompt}])
        verdict = "".join(b.text for b in r.content if b.type == "text").strip()
        if verdict:
            return verdict.lstrip("*# ").upper().startswith("PASS"), verdict
    return False, "ERROR: judge returned no verdict"

def eval_info_agent():
    print("\n=== C. INFO AGENT BEHAVIOR ===")
    passed = total = 0
    for case in INFO_CASES:
        reply = info_agent({"messages": [u(case["q"])]})["messages"][-1]["content"]
        low = reply.lower()
        checks = []
        if case["answerable"]:
            checks.append(("cites a source", ".txt" in reply))
            ok, why = judge_faithful(case["q"], reply)
            checks.append((f"faithful (judge: {why[:70]})", ok))
        else:
            checks.append(("says it doesn't know", any(p in low for p in UNKNOWN_PHRASES)))
        if case["allergy"]:
            checks.append(("has allergy safety line", "severe allergy" in low))
        print(f"\n  Q: {case['q']}")
        for name, ok in checks:
            print(f"     {'PASS' if ok else 'FAIL'}  {name}")
            passed += ok
            total += 1
    return f"{passed}/{total} checks passed"

# ================= SCORECARD =================
r1, r2, r3 = eval_retrieval(), eval_router(), eval_info_agent()
print("\n=========== EVAL SCORECARD ===========")
print("  Retrieval :", r1)
print("  Router    :", r2)
print("  Info agent:", r3)
