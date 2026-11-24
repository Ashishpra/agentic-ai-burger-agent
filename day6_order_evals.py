import io, re, time, contextlib

# Load the real order_agent from day4.py (without starting its chat loop)
src = open("day4.py").read().split("# ---------- App loop")[0]
ns = {}
exec(src, ns)
order_agent = ns["order_agent"]

# ---------- Count cost per model (list prices per million tokens, Nov 2026; check current pricing) ----------
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50)}
COST = {}

class CountingMessages:
    def __init__(self, real):
        self.real = real
    def create(self, **kwargs):
        r = self.real.create(**kwargs)
        pin, pout = PRICES.get(kwargs["model"], (0, 0))
        COST[kwargs["model"]] = COST.get(kwargs["model"], 0) + (
            r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        return r

class CountingClient:
    def __init__(self, real):
        self.messages = CountingMessages(real.messages)

ns["client"] = CountingClient(ns["client"])

def u(t): return {"role": "user", "content": t}
def a(t): return {"role": "assistant", "content": t}

BURGER = {"item": "classic burger", "size": "regular", "qty": 1}
LARGE_FRIES = {"item": "fries", "size": "large", "qty": 1}
USUAL = [("classic burger", "regular"), ("coke", "medium"), ("fries", "large")]

def items(cart):
    return sorted((c["item"], c["size"]) for c in cart)

# ---------- ORDER-AGENT CASES: graded by CODE (cart, tools, checkout flag), from the Step 1 red lines ----------
CASES = [
    ("Asks for a size when it's missing",
     [u("add some fries")], [],
     lambda r: r["cart"] == [] and not r["pending"] and "size" in r["reply"].lower()),
    ("Adds an item without checking out",
     [u("add a classic burger")], [],
     lambda r: items(r["cart"]) == [("classic burger", "regular")] and not r["pending"]),
    ("'The usual' uses order history, never adds wrong items",
     [u("gimme the usual")], [],
     lambda r: "get_order_history" in r["tools"] and items(r["cart"]) in (USUAL, []) and not r["pending"]),
    ("Total comes from a tool and is correct ($10.43)",
     [u("what's my total?")], [BURGER, LARGE_FRIES],
     lambda r: len(r["tools"]) > 0 and "10.43" in r["reply"]
     and r["cart"] == [BURGER, LARGE_FRIES] and not r["pending"]),
    ("Rejects an off-menu item",
     [u("add a vanilla shake")], [],
     lambda r: r["cart"] == [] and not r["pending"]),
    ("Requests checkout when the customer confirms",
     [u("add a classic burger"), a("I added a classic burger. Your total is $6.59. Should I place your order?"),
      u("yes, place it")], [BURGER],
     lambda r: r["pending"] and r["cart"] == [BURGER]),
]

MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5"]
RUNS = 3

def run_case(model, msgs, cart):
    ns["MODEL"] = model
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):          # capture the [tool] lines the agent prints
        out = order_agent({"messages": msgs, "cart": [dict(c) for c in cart], "orders": [],
                           "route": "order", "pending_checkout": False})
    return {"cart": out["cart"], "pending": out["pending_checkout"],
            "reply": out["messages"][-1]["content"],
            "tools": re.findall(r"\[tool\] (\w+)\(", buf.getvalue()),
            "ms": (time.perf_counter() - t0) * 1000}

summary = {}
for model in MODELS:
    short = model.replace("claude-", "")
    print(f"\n=== ORDER AGENT on {short} ({RUNS} runs per case) ===")
    passed, total, times = 0, 0, []
    for name, msgs, cart, check in CASES:
        results = [run_case(model, msgs, cart) for _ in range(RUNS)]
        oks = [check(r) for r in results]
        passed += sum(oks)
        total += len(oks)
        times += [r["ms"] for r in results]
        print(f"  {'PASS' if all(oks) else 'FAIL'}  {sum(oks)}/{RUNS}  {name}")
        for r, ok in zip(results, oks):
            if not ok:
                print(f"          tools={r['tools']} pending={r['pending']} cart={items(r['cart'])}")
                print(f"          reply: {r['reply'][:120]!r}")
    summary[short] = (passed, total, sum(times) / len(times), COST.get(model, 0))

print("\n=========== ORDER AGENT: SONNET vs HAIKU ===========")
print(f"  {'model':<12} {'passed':>9} {'avg latency':>12} {'eval cost':>10}")
for short, (p, t, ms, cost) in summary.items():
    print(f"  {short:<12} {f'{p}/{t}':>9} {f'{ms:,.0f} ms':>12} {f'${cost:.4f}':>10}")
