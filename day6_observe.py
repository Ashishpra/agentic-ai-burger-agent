import io, time, contextlib
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Reuse the real nodes from day4.py (without starting its chat loop)
src = open("day4.py").read().split("# ---------- App loop")[0]
ns = {}
exec(src, ns)

# List prices in USD per million tokens (input, output), from Anthropic's pricing page, Nov 2026.
# Prices change: always check the current pricing page before relying on these numbers.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}

SPANS, STACK, TURN = [], [], {"n": 0}

def cost_of(model, tin, tout):
    pin, pout = PRICES.get(model, (0, 0))
    return tin * pin / 1e6 + tout * pout / 1e6

# ---------- Span 1: every LLM call (wrap the client the nodes use) ----------
class TracedMessages:
    def __init__(self, real):
        self.real = real
    def create(self, **kwargs):
        t0 = time.perf_counter()
        r = self.real.create(**kwargs)
        model = kwargs.get("model", "?")
        SPANS.append({"turn": TURN["n"], "kind": "llm", "node": STACK[-1] if STACK else "?",
                      "model": model, "in": r.usage.input_tokens, "out": r.usage.output_tokens,
                      "ms": (time.perf_counter() - t0) * 1000,
                      "cost": cost_of(model, r.usage.input_tokens, r.usage.output_tokens)})
        return r

class TracedClient:
    def __init__(self, real):
        self.messages = TracedMessages(real.messages)

ns["client"] = TracedClient(ns["client"])

# ---------- Span 2: tool calls (RAG search) ----------
real_search = ns["search_store_info"]
def traced_search(question, **kw):
    t0 = time.perf_counter()
    out = real_search(question, **kw)
    SPANS.append({"turn": TURN["n"], "kind": "tool", "node": STACK[-1] if STACK else "?",
                  "name": "search_store_info", "results": len(out.get("results", [])),
                  "ms": (time.perf_counter() - t0) * 1000})
    return out
ns["search_store_info"] = traced_search

# ---------- Span 3: every graph node (parent of the LLM and tool spans inside it) ----------
def traced(name, fn):
    def wrapper(state):
        span = {"turn": TURN["n"], "kind": "node", "name": name, "ms": None}
        SPANS.append(span)
        STACK.append(name)
        t0 = time.perf_counter()
        try:
            return fn(state)
        finally:
            STACK.pop()
            span["ms"] = (time.perf_counter() - t0) * 1000
    return wrapper

b = StateGraph(ns["State"])
for name in ["supervisor", "order_agent", "info_agent", "fallback", "approval"]:
    b.add_node(name, traced(name, ns[name]))
b.add_edge(START, "supervisor")
b.add_conditional_edges("supervisor", lambda s: s["route"],
                        {"order": "order_agent", "info": "info_agent", "other": "fallback"})
b.add_conditional_edges("order_agent", lambda s: "approval" if s["pending_checkout"] else "done",
                        {"approval": "approval", "done": END})
b.add_edge("info_agent", END)
b.add_edge("fallback", END)
b.add_edge("approval", END)
graph = b.compile(checkpointer=InMemorySaver())

def print_trace(i, text, total_ms, result):
    spans = [s for s in SPANS if s["turn"] == i]
    cost = sum(s.get("cost", 0) for s in spans)
    print(f'\nTRACE turn {i}: "{text}"   total {total_ms:,.0f} ms, ${cost:.5f}')
    for s in spans:
        if s["kind"] == "node":
            print(f"  ├─ {s['name']:<12} {s['ms']:>7,.0f} ms")
        elif s["kind"] == "llm":
            print(f"  │    └─ llm  {s['model'].replace('claude-', ''):<10} in {s['in']:>6,} out {s['out']:>5,}"
                  f"  {s['ms']:>6,.0f} ms  ${s['cost']:.5f}")
        else:
            print(f"  │    └─ tool {s['name']} ({s['results']} results)  {s['ms']:>6,.0f} ms")
    print(f"  reply: {result['messages'][-1]['content'][:110]!r}")

# ---------- Run a realistic 6-turn conversation ----------
TURNS = ["does the swirl have nuts?", "ok add a swirl and a classic burger", "add some fries",
         "large", "what's my total?", "how do I get a free burger?"]
config = {"configurable": {"thread_id": "observe-1"}}

for i, text in enumerate(TURNS, 1):
    TURN["n"] = i
    prev = graph.get_state(config).values or {}
    state_in = {"messages": prev.get("messages", []) + [{"role": "user", "content": text}],
                "cart": prev.get("cart", []), "orders": prev.get("orders", []),
                "route": "", "pending_checkout": False}
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):      # hide the nodes' own print lines
        result = graph.invoke(state_in, config)
    print_trace(i, text, (time.perf_counter() - t0) * 1000, result)

# ---------- COST REPORT ----------
llm = [s for s in SPANS if s["kind"] == "llm"]
total = sum(s["cost"] for s in llm)
print("\n=========== COST BY NODE ===========")
print(f"  {'node':<12} {'calls':>5} {'in tokens':>10} {'out tokens':>10} {'cost':>10} {'share':>6}")
for node in dict.fromkeys(s["node"] for s in llm):
    ss = [s for s in llm if s["node"] == node]
    c = sum(s["cost"] for s in ss)
    print(f"  {node:<12} {len(ss):>5} {sum(s['in'] for s in ss):>10,} {sum(s['out'] for s in ss):>10,}"
          f" {'$' + format(c, '.5f'):>10} {c / total * 100:>5.0f}%")

print("\n=========== INPUT TOKENS PER TURN (history is resent every turn) ===========")
for i in range(1, len(TURNS) + 1):
    tin = sum(s["in"] for s in llm if s["turn"] == i)
    print(f"  turn {i}: {tin:>7,} input tokens  {'#' * max(1, tin // 250)}")

print("\n=========== SCALE ===========")
print(f"  This {len(TURNS)}-turn conversation: ${total:.4f} across {len(llm)} LLM calls")
print(f"  At 1,000,000 conversations per month: about ${total * 1_000_000:,.0f} per month")
