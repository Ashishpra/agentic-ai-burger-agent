import anthropic, json, itertools
from rag import search_store_info

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
TAX_RATE = 0.10

# ---------- Backend data ----------
MENU = {
    "classic burger":  {"regular": 5.99},
    "swirl": {"regular": 4.29},
    "fries":    {"small": 2.49, "medium": 2.99, "large": 3.49},
    "coke":     {"small": 1.49, "medium": 1.99, "large": 2.29},
}
ORDER_HISTORY = {"C123": [["Classic Burger", "large fries", "medium coke"],
                          ["Classic Burger", "large fries", "medium coke"],
                          ["Swirl"]]}
CURRENT_CUSTOMER = "C123"        # from the login session, NEVER from the model

# ---------- Session state lives in CODE, not in the LLM ----------
CART = []
PLACED_ORDERS = []               # source of truth for placed orders
order_ids = itertools.count(1001)

def find_item(name):
    for key in MENU:
        if key in name.lower():
            return key
    return None

def get_menu_price(item, size="regular"):
    key = find_item(item)
    if not key:
        return {"error": f"'{item}' is not on our menu"}
    if size not in MENU[key]:
        return {"error": f"'{key}' comes in sizes: {list(MENU[key])}"}
    return {"item": key, "size": size, "price": MENU[key][size]}

def calculate_total(items):
    lines, subtotal = [], 0
    for it in items:
        p = get_menu_price(it["item"], it.get("size", "regular"))
        qty = it.get("qty", 1)
        lines.append({"item": p["item"], "size": p["size"], "qty": qty,
                      "line_total": round(p["price"] * qty, 2)})
        subtotal += p["price"] * qty
    tax = round(subtotal * TAX_RATE, 2)
    return {"lines": lines, "subtotal": round(subtotal, 2), "tax": tax,
            "total": round(subtotal + tax, 2)}

# ---------- Tools ----------
def get_order_history():
    return {"recent_orders": ORDER_HISTORY.get(CURRENT_CUSTOMER, [])}

def view_cart():
    if not CART:
        return {"cart": "empty", "total": 0}
    return calculate_total(CART)

def add_to_cart(items):
    for it in items:                              # validate ALL items first
        p = get_menu_price(it["item"], it.get("size", "regular"))
        if "error" in p:
            return p
    for it in items:
        p = get_menu_price(it["item"], it.get("size", "regular"))
        CART.append({"item": p["item"], "size": p["size"], "qty": it.get("qty", 1)})
    return view_cart()

def remove_from_cart(item, size="regular"):
    key = find_item(item)
    for i, c in enumerate(CART):
        if c["item"] == key and c["size"] == size:
            CART.pop(i)
            return view_cart()
    return {"error": f"'{item}' ({size}) is not in the cart"}

def checkout():                                    # no parameters on purpose
    if not CART:
        return {"error": "cart is empty, nothing to place"}
    bill = calculate_total(CART)
    answer = input(f"\n   >>> [APP CONFIRM BUTTON] place order for ${bill['total']}? (y/n): ")
    if answer.strip().lower() not in ("y", "yes"):
        return {"status": "cancelled", "reason": "customer did not confirm", "cart_kept": True}
    order = {"order_id": f"ORD-{next(order_ids)}", "items": list(CART), "total": bill["total"]}
    PLACED_ORDERS.append(order)
    CART.clear()
    return {"status": "placed", **order}

def get_placed_orders():
    return {"orders": PLACED_ORDERS}

SIZE = {"type": "string", "enum": ["small", "medium", "large", "regular"]}
ITEMS = {"type": "array", "items": {"type": "object", "properties": {
    "item": {"type": "string"}, "size": SIZE, "qty": {"type": "integer"}},
    "required": ["item"]}}
EMPTY = {"type": "object", "properties": {}}

TOOLS = [
    {"name": "search_store_info", "description": "Search store documents: allergens, nutrition, hours, refunds, loyalty. Returns source passages to cite.", "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
    {"name": "get_order_history", "input_schema": EMPTY,
     "description": "Customer's recent orders. Use for 'the usual' or 'same as last time'."},
    {"name": "get_menu_price", "description": "Price of ONE item and size.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string"}, "size": SIZE},
                      "required": ["item"]}},
    {"name": "add_to_cart", "description": "Add items to the cart. Does NOT place an order.",
     "input_schema": {"type": "object", "properties": {"items": ITEMS}, "required": ["items"]}},
    {"name": "remove_from_cart", "description": "Remove one item from the cart.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string"}, "size": SIZE},
                      "required": ["item"]}},
    {"name": "view_cart", "input_schema": EMPTY,
     "description": "Show cart items, subtotal, tax and total. Use for any total question."},
    {"name": "checkout", "input_schema": EMPTY,
     "description": "Place the order for everything in the cart. ONLY when the customer clearly "
                    "asks to place, checkout or pay. The app asks the customer to confirm."},
    {"name": "get_placed_orders", "input_schema": EMPTY,
     "description": "Orders already placed in this session (source of truth)."},
]

def run_tool(name, args):
    fns = {"get_order_history": get_order_history, "get_menu_price": get_menu_price,
           "add_to_cart": add_to_cart, "remove_from_cart": remove_from_cart,
           "view_cart": view_cart, "checkout": checkout, "get_placed_orders": get_placed_orders, "search_store_info": search_store_info}
    if name not in fns:
        return {"error": f"unknown tool {name}"}
    return fns[name](**args)

SYSTEM = ("You are a friendly ordering assistant for a burger restaurant. "
          "Customers build a cart: use add_to_cart for requested items. Never checkout unless "
          "the customer clearly asks to place, checkout or pay. Totals come only from view_cart; "
          "never do math yourself. If a fries or coke size is missing, ask. For allergens, nutrition, hours, refunds or loyalty questions, use search_store_info and cite the source file name. If it returns no results, say you do not know. For any allergy question, end with: Please confirm with staff if you have a severe allergy. Keep replies short.")

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

messages = []
print("Burger Agent v2 (with cart). Type 'quit' to exit.")
while True:
    user = input("\nYOU: ")
    if user.strip().lower() in ("quit", "exit"):
        break
    if not user.strip():
        continue
    messages.append({"role": "user", "content": user})
    print("AGENT:", agent_turn(messages))
    v = view_cart()                                  # the APP shows totals, from code
    print(f"   [APP CART] {len(CART)} line(s), total ${v['total']} | placed orders: {len(PLACED_ORDERS)}")
