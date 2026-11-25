"""Burger shop MCP server: exposes the burger tools over the Model Context Protocol (stdio transport).

IMPORTANT for stdio servers: stdout carries the protocol messages, so never print() to stdout here.
Log to stderr instead, or a stray print will corrupt the connection.
"""
import itertools, json, sys
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("burger-shop")

# FICTIONAL data for learning only
TAX_RATE = 0.10
MENU = {
    "classic burger": {"regular": 5.99},
    "swirl":          {"regular": 4.29},
    "fries":          {"small": 2.49, "medium": 2.99, "large": 3.49},
    "coke":           {"small": 1.49, "medium": 1.99, "large": 2.29},
}
CART, ORDERS = [], []
order_ids = itertools.count(1001)

def log(msg):
    print(f"[server] {msg}", file=sys.stderr)   # stderr is safe; stdout is the protocol

def _find(name):
    for key in MENU:
        if key in name.lower():
            return key
    return None

def _price(item, size):
    key = _find(item)
    if not key:
        return {"error": f"'{item}' is not on our menu"}
    if size not in MENU[key]:
        return {"error": f"'{key}' comes in sizes: {list(MENU[key])}"}
    return {"item": key, "size": size, "price": MENU[key][size]}

def _total(items):
    lines, subtotal = [], 0.0
    for it in items:
        p = MENU[it["item"]][it["size"]]
        lines.append({**it, "line_total": round(p * it["qty"], 2)})
        subtotal += p * it["qty"]
    tax = round(subtotal * TAX_RATE, 2)
    return {"lines": lines, "subtotal": round(subtotal, 2), "tax": tax, "total": round(subtotal + tax, 2)}

# ---------- TOOLS: actions a model can call ----------
@mcp.tool()
def get_menu_price(item: str, size: str = "regular") -> dict:
    """Price of ONE menu item and size. Sizes: small, medium, large, or regular for items without sizes."""
    return _price(item, size)

@mcp.tool()
def add_to_cart(item: str, size: str = "regular", qty: int = 1) -> dict:
    """Add ONE item to the cart. Does NOT place an order. Fries and coke need a size: ask the customer if missing."""
    p = _price(item, size)
    if "error" in p:
        return p
    CART.append({"item": p["item"], "size": p["size"], "qty": qty})
    log(f"cart now has {len(CART)} line(s)")
    return _total(CART)

@mcp.tool()
def view_cart() -> dict:
    """Show cart items, subtotal, tax and total. Use for any total question."""
    return _total(CART) if CART else {"cart": "empty", "total": 0}

@mcp.tool()
def checkout() -> dict:
    """Place the order for everything in the cart. Use ONLY when the customer clearly asks to place, checkout or pay."""
    if not CART:
        return {"error": "cart is empty, nothing to place"}
    order = {"order_id": f"ORD-{next(order_ids)}", "items": list(CART), "total": _total(CART)["total"]}
    ORDERS.append(order)
    CART.clear()
    log(f"placed {order['order_id']}")
    return {"status": "placed", **order}

@mcp.tool()
def search_store_info(question: str) -> dict:
    """Search store documents: allergens, nutrition, hours, refunds, loyalty. Cite the source file in answers."""
    from rag import search_store_info as rag_search   # lazy import: loads the RAG models on first use
    out = rag_search(question)
    if any(r["source"] == "allergen_guide.txt" for r in out.get("results", [])):
        out["safety_note"] = "Please confirm with staff if you have a severe allergy."
    return out

# ---------- RESOURCE: read-only data an app can load ----------
@mcp.resource("menu://items")
def menu() -> str:
    """The full menu with prices."""
    return json.dumps(MENU, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")   # stdio transport: the host starts this process and talks over stdin/stdout
