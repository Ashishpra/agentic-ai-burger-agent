import anthropic, json

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
TAX_RATE = 0.10

# ---------- Fake backend: now size-aware ----------
MENU = {
    "classic burger":  {"regular": 5.99},
    "swirl": {"regular": 4.29},
    "fries":    {"small": 2.49, "medium": 2.99, "large": 3.49},
    "coke":     {"small": 1.49, "medium": 1.99, "large": 2.29},
}
ORDER_HISTORY = {"C123": [["Classic Burger", "large fries", "medium coke"],
                          ["Classic Burger", "large fries", "medium coke"],
                          ["Swirl"]]}
CURRENT_CUSTOMER = "C123"   # from the login session, NEVER from the model

def find_item(name):
    for key in MENU:
        if key in name.lower():
            return key
    return None

def get_order_history():
    return {"recent_orders": ORDER_HISTORY.get(CURRENT_CUSTOMER, [])}

def get_menu_price(item, size="regular"):
    key = find_item(item)
    if not key:
        return {"error": f"'{item}' is not on our menu"}
    if size not in MENU[key]:
        return {"error": f"'{key}' comes in sizes: {list(MENU[key])}"}
    return {"item": key, "size": size, "price": MENU[key][size]}

def calculate_total(items):          # deterministic money math
    lines, subtotal = [], 0
    for it in items:
        p = get_menu_price(it["item"], it.get("size", "regular"))
        if "error" in p:
            return p
        qty = it.get("qty", 1)
        lines.append({"item": p["item"], "size": p["size"], "qty": qty,
                      "line_total": round(p["price"] * qty, 2)})
        subtotal += p["price"] * qty
    tax = round(subtotal * TAX_RATE, 2)
    return {"lines": lines, "subtotal": round(subtotal, 2), "tax": tax,
            "total": round(subtotal + tax, 2)}

def place_order(items):              # WRITE action -> human approval gate in code
    bill = calculate_total(items)
    if "error" in bill:
        return bill
    answer = input(f"\n   >>> APPROVAL NEEDED: place order for ${bill['total']}? (y/n): ")
    if answer.strip().lower() != "y":
        return {"status": "cancelled", "reason": "customer did not confirm"}
    return {"status": "placed", "order_id": "ORD-1001", "total": bill["total"]}

# ---------- Tool definitions ----------
ITEMS = {"type": "array", "items": {"type": "object", "properties": {
            "item": {"type": "string"},
            "size": {"type": "string", "enum": ["small", "medium", "large", "regular"]},
            "qty":  {"type": "integer"}}, "required": ["item"]}}

TOOLS = [
    {"name": "get_order_history",
     "description": "Get the logged-in customer's recent orders. Use for 'the usual' or 'same as last time'.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_menu_price",
     "description": "Price of ONE item and size. Items without sizes use 'regular'.",
     "input_schema": {"type": "object", "properties": {
         "item": {"type": "string"},
         "size": {"type": "string", "enum": ["small", "medium", "large", "regular"]}},
         "required": ["item"]}},
    {"name": "calculate_total",
     "description": "Calculate subtotal, tax and total for a list of items. ALWAYS use this for totals.",
     "input_schema": {"type": "object", "properties": {"items": ITEMS}, "required": ["items"]}},
    {"name": "place_order",
     "description": "Place the order. The system itself asks the customer to approve before charging.",
     "input_schema": {"type": "object", "properties": {"items": ITEMS}, "required": ["items"]}},
]

def run_tool(name, args):
    if name == "get_order_history": return get_order_history()
    if name == "get_menu_price":    return get_menu_price(**args)
    if name == "calculate_total":   return calculate_total(**args)
    if name == "place_order":       return place_order(**args)
    return {"error": f"unknown tool {name}"}

SYSTEM = ("You are a friendly ordering assistant for a burger restaurant. "
          "Use tools for history, prices and totals. Never do math yourself; use calculate_total. "
          "If a fries or coke size is missing, ask the customer. Keep replies short.")

# ---------- Agent loop for one customer turn ----------
def agent_turn(messages, max_steps=6):
    for _ in range(max_steps):
        r = client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM,
                                   tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text")
        results = []
        for b in r.content:
            if b.type == "tool_use":
                out = run_tool(b.name, b.input)
                print(f"   [tool] {b.name}({json.dumps(b.input)}) -> {json.dumps(out)}")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    fallback = "Sorry, I couldn't finish that. Let me get a team member."
    messages.append({"role": "assistant", "content": fallback})
    return fallback

# ---------- Chat loop: conversation memory lives HERE, in your app ----------
messages = []
print("Burger Agent ready. Type 'quit' to exit.")
while True:
    user = input("\nYOU: ")
    if user.strip().lower() in ("quit", "exit"):
        break
    messages.append({"role": "user", "content": user})
    print("AGENT:", agent_turn(messages))
