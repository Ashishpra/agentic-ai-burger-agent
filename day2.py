import anthropic, json

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

# ---------- Fake backend (in real life: your company APIs) ----------
MENU = {"classic burger": 5.99, "fries": 3.49, "coke": 1.99, "swirl": 4.29}
ORDER_HISTORY = {"C123": [["Classic Burger", "large fries", "medium coke"],
                          ["Classic Burger", "large fries", "medium coke"],
                          ["Swirl"]]}
CURRENT_CUSTOMER = "C123"   # comes from the login session, NEVER from the model

def get_order_history():
    return {"recent_orders": ORDER_HISTORY.get(CURRENT_CUSTOMER, [])}

def get_menu_price(item):
    for name, price in MENU.items():
        if name in item.lower():
            return {"item": name, "price": price}
    return {"error": f"'{item}' is not on our menu"}

# ---------- Tool definitions: the "API spec" the model reads ----------
TOOLS = [
    {"name": "get_order_history",
     "description": "Get the logged-in customer's recent orders. Use when they say 'the usual' or 'same as last time'.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_menu_price",
     "description": "Look up the current price of ONE menu item. Never guess prices.",
     "input_schema": {"type": "object",
                      "properties": {"item": {"type": "string"}},
                      "required": ["item"]}},
]

def run_tool(name, args):
    if name == "get_order_history": return get_order_history()
    if name == "get_menu_price":    return get_menu_price(**args)
    return {"error": f"unknown tool {name}"}

SYSTEM = ("You are a friendly ordering assistant for a burger restaurant. "
          "Use tools for order history and prices; never guess. Keep replies short.")

# ---------- The agent loop (ReAct) ----------
def agent(user_text, max_steps=5):
    messages = [{"role": "user", "content": user_text}]
    for step in range(max_steps):
        r = client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM,
                                   tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": r.content})

        if r.stop_reason != "tool_use":          # model is done
            return "".join(b.text for b in r.content if b.type == "text")

        results = []
        for b in r.content:                       # model asked for tools
            if b.type == "tool_use":
                out = run_tool(b.name, b.input)
                print(f"   [tool] {b.name}({b.input}) -> {out}")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})

    return "Sorry, I couldn't finish that. Let me get a team member."   # HITL fallback

tests = [
    "gimme the usual",
    "how much is a classic burger?",
    "gimme the usual plus a swirl. what's my total?",
    "how much is a whopper?",
]
for q in tests:
    print("\nCUSTOMER:", q)
    print("AGENT:", agent(q))
