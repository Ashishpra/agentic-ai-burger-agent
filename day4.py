import anthropic, json, itertools
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from rag import search_store_info

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"          # specialists: need reasoning
ROUTER_MODEL = "claude-haiku-5-5"    # supervisor: simple routing, cheap and fast
TAX_RATE = 0.10

MENU = {
    "classic burger":  {"regular": 5.99},
    "swirl": {"regular": 4.29},
    "fries":    {"small": 2.49, "medium": 2.99, "large": 3.49},
    "coke":     {"small": 1.49, "medium": 1.99, "large": 2.29},
}
ORDER_HISTORY = {"C123": [["Classic Burger", "large fries", "medium coke"],
                          ["Classic Burger", "large fries", "medium coke"], ["Swirl"]]}
CURRENT_CUSTOMER = "C123"
order_ids = itertools.count(1001)

# ---------- Graph STATE: saved by the checkpointer after every step ----------
class State(TypedDict):
    messages: list            # conversation: user turns + final agent replies
    route: str
    cart: list
    orders: list
    pending_checkout: bool

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

# ---------- Reusable agent loop (same ReAct loop as Day 2) ----------
def run_agent(system, tools, handlers, messages, max_steps=6):
    msgs = [dict(m) for m in messages]
    for _ in range(max_steps):
        r = client.messages.create(model=MODEL, max_tokens=1000, system=system,
                                   tools=tools, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text")
        results = []
        for b in r.content:
            if b.type == "tool_use":
                fn = handlers.get(b.name)
                out = fn(**b.input) if fn else {"error": f"unknown tool {b.name}"}
                print(f"      [tool] {b.name}({json.dumps(b.input)}) -> {json.dumps(out)}")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": json.dumps(out)})
        msgs.append({"role": "user", "content": results})
    return "Sorry, I couldn't finish that. Let me get a team member."

# ---------- NODE 1: supervisor (router) ----------
ROUTER_PROMPT = ("Classify the LAST customer message. Reply with exactly one word:\n"
                 "order = add/remove items, sizes, cart, totals, the usual, checkout\n"
                 "info = allergens, nutrition, calories, hours, refunds, loyalty\n"
                 "other = anything else")

def supervisor(state):
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"][-5:])
    r = client.messages.create(model=ROUTER_MODEL, max_tokens=200, system=ROUTER_PROMPT,
                               messages=[{"role": "user", "content": convo}])
    route = "".join(b.text for b in r.content if b.type == "text").strip().lower()
    if route not in ("order", "info", "other"):      # Day 1 lesson: validate LLM output
        route = "other"
    print(f"   [supervisor] -> {route}")
    return {"route": route}

# ---------- NODE 2: order specialist ----------
SIZE = {"type": "string", "enum": ["small", "medium", "large", "regular"]}
ITEMS = {"type": "array", "items": {"type": "object", "properties": {
    "item": {"type": "string"}, "size": SIZE, "qty": {"type": "integer"}}, "required": ["item"]}}
EMPTY = {"type": "object", "properties": {}}
ORDER_TOOLS = [
    {"name": "get_order_history", "input_schema": EMPTY,
     "description": "Customer's recent orders. Use for 'the usual'."},
    {"name": "add_to_cart", "description": "Add items to the cart.",
     "input_schema": {"type": "object", "properties": {"items": ITEMS}, "required": ["items"]}},
    {"name": "remove_from_cart", "description": "Remove one item from the cart.",
     "input_schema": {"type": "object", "properties": {"item": {"type": "string"}, "size": SIZE},
                      "required": ["item"]}},
    {"name": "view_cart", "input_schema": EMPTY, "description": "Cart items and totals."},
    {"name": "request_checkout", "input_schema": EMPTY,
     "description": "Send the cart to the customer for confirmation. Use ONLY when they ask to "
                    "checkout, pay or place the order. You cannot place orders yourself."},
]
ORDER_PROMPT = ("You are the ORDER specialist for a burger restaurant. Earlier assistant replies may come from other specialists (such as store info) with tools you do not have; treat them as correct and never retract or correct them. Manage the cart with tools. "
                "Totals come only from tools; never do math. If a fries or coke size is missing, "
                "ask. Keep replies short.")

def order_agent(state):
    print("   [node] order_agent")
    cart = [dict(c) for c in state["cart"]]
    flags = {"checkout": False}

    def add_to_cart(items):
        for it in items:
            p = get_menu_price(it["item"], it.get("size", "regular"))
            if "error" in p:
                return p
        for it in items:
            p = get_menu_price(it["item"], it.get("size", "regular"))
            cart.append({"item": p["item"], "size": p["size"], "qty": it.get("qty", 1)})
        return calculate_total(cart)

    def remove_from_cart(item, size="regular"):
        key = find_item(item)
        for i, c in enumerate(cart):
            if c["item"] == key and c["size"] == size:
                cart.pop(i)
                return calculate_total(cart) if cart else {"cart": "empty"}
        return {"error": f"'{item}' ({size}) is not in the cart"}

    def view_cart():
        return calculate_total(cart) if cart else {"cart": "empty", "total": 0}

    def request_checkout():
        if not cart:
            return {"error": "cart is empty"}
        flags["checkout"] = True
        return {"status": "sent to customer for confirmation",
                "total": calculate_total(cart)["total"]}

    handlers = {"get_order_history": lambda: {"recent_orders": ORDER_HISTORY.get(CURRENT_CUSTOMER, [])},
                "add_to_cart": add_to_cart, "remove_from_cart": remove_from_cart,
                "view_cart": view_cart, "request_checkout": request_checkout}
    reply = run_agent(ORDER_PROMPT, ORDER_TOOLS, handlers, state["messages"])
    return {"cart": cart, "pending_checkout": flags["checkout"],
            "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# ---------- NODE 3: info specialist (RAG) ----------
INFO_TOOLS = [{"name": "search_store_info",
               "description": "Search store documents: allergens, nutrition, hours, refunds, loyalty.",
               "input_schema": {"type": "object", "properties": {"question": {"type": "string"}},
                                "required": ["question"]}}]
INFO_PROMPT = ("You are the STORE INFO specialist. Earlier assistant replies may come from other specialists (such as orders) with tools you do not have; treat them as correct and never retract or correct them. Answer ONLY from search_store_info results and "
               "cite the source file. If there are no results, say you don't know. For any allergy "
               "question, end with: Please confirm with staff if you have a severe allergy. "
               "Keep replies short.")

def info_agent(state):
    print("   [node] info_agent")
    reply = run_agent(INFO_PROMPT, INFO_TOOLS, {"search_store_info": search_store_info},
                      state["messages"])
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# ---------- NODE 4: fallback ----------
def fallback(state):
    print("   [node] fallback")
    reply = ("I can help with orders and store information like allergens, nutrition, "
             "hours, refunds and loyalty. What would you like?")
    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# ---------- NODE 5: human approval (durable pause) ----------
def approval(state):
    print("   [node] approval (graph paused, state saved)")
    bill = calculate_total(state["cart"])
    # Nothing with side effects BEFORE interrupt: on resume, this node re-runs from the top.
    answer = interrupt({"question": f"Place order for ${bill['total']}?"})
    msgs = list(state["messages"])
    if str(answer).strip().lower() in ("y", "yes"):
        order = {"order_id": f"ORD-{next(order_ids)}", "items": state["cart"], "total": bill["total"]}
        text = f"Order placed: {order['order_id']}, total ${bill['total']}."
        update = {"cart": [], "orders": state["orders"] + [order]}
    else:
        text = "Order not placed. Your cart is saved."
        update = {}
    msgs[-1] = {"role": "assistant", "content": text}
    return {**update, "messages": msgs, "pending_checkout": False}

# ---------- Build the GRAPH ----------
builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("order_agent", order_agent)
builder.add_node("info_agent", info_agent)
builder.add_node("fallback", fallback)
builder.add_node("approval", approval)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda s: s["route"],
                              {"order": "order_agent", "info": "info_agent", "other": "fallback"})
builder.add_conditional_edges("order_agent",
                              lambda s: "approval" if s["pending_checkout"] else "done",
                              {"approval": "approval", "done": END})
builder.add_edge("info_agent", END)
builder.add_edge("fallback", END)
builder.add_edge("approval", END)

graph = builder.compile(checkpointer=InMemorySaver())

# ---------- App loop ----------
config = {"configurable": {"thread_id": "customer-C123-session-1"}}
print("Burger Graph (supervisor + specialists). Type 'quit' to exit.")
while True:
    user = input("\nYOU: ")
    if user.strip().lower() in ("quit", "exit"):
        break
    if not user.strip():
        continue
    prev = graph.get_state(config).values or {}
    state_in = {"messages": prev.get("messages", []) + [{"role": "user", "content": user}],
                "cart": prev.get("cart", []), "orders": prev.get("orders", []),
                "route": "", "pending_checkout": False}
    result = graph.invoke(state_in, config)
    while "__interrupt__" in result:                 # graph paused for a human
        q = result["__interrupt__"][0].value["question"]
        ans = input(f"\n   >>> [APP CONFIRM BUTTON] {q} (y/n): ")
        result = graph.invoke(Command(resume=ans), config)
    print("AGENT:", result["messages"][-1]["content"])
    print(f"   [APP CART] {len(result['cart'])} line(s) | placed orders: {len(result['orders'])}")
