"""Burger chat: a local web chat UI for your MCP-powered agent.

Run:   python3 chat_app.py      then open  http://localhost:8000
Uses only the Python standard library for the web part, plus anthropic and mcp (already installed).
"""
import asyncio, json, sys, threading, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "claude-haiku-5-5"
NEEDS_APPROVAL = {"checkout"}
SYSTEM = ("You are a friendly ordering assistant for a burger restaurant. Use the tools for prices, the cart "
          "and store information; never guess or do math yourself. If fries or coke size is missing, ask. "
          "Only call checkout when the customer clearly asks to place the order. Keep replies short.")
SERVER = StdioServerParameters(command=sys.executable, args=["burger_mcp_server.py"])

client = anthropic.Anthropic()
LOOP = asyncio.new_event_loop()
STATE = {"session": None, "tools": [], "messages": [], "pending": None}
READY = threading.Event()
LOCK = threading.Lock()

# ---------- Keep one MCP session open in a background thread ----------
async def mcp_main():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            STATE["tools"] = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                              for t in listed.tools]
            STATE["session"] = session
            READY.set()
            await asyncio.Event().wait()          # stay connected until the app stops

threading.Thread(target=lambda: LOOP.run_until_complete(mcp_main()), daemon=True).start()
if not READY.wait(60):
    sys.exit("Could not connect to the MCP server (burger_mcp_server.py). Is it in this folder?")
print("Connected to MCP server. Tools:", [t["name"] for t in STATE["tools"]])

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, LOOP).result(timeout=120)

def call_tool(name, args):
    res = run_async(STATE["session"].call_tool(name, args))
    return "".join(c.text for c in res.content if c.type == "text")

def exec_tool(b, trace):
    text = call_tool(b.name, b.input)
    trace.append({"tool": b.name, "input": b.input, "output": text[:400]})
    return {"type": "tool_result", "tool_use_id": b.id, "content": text}

def cart_snapshot():
    try:
        return json.loads(call_tool("view_cart", {}))
    except Exception:
        return None

# ---------- The agent loop, pausing for approval instead of blocking ----------
def agent_continue(trace):
    for _ in range(6):
        r = client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM,
                                   tools=STATE["tools"], messages=STATE["messages"])
        STATE["messages"].append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            return {"reply": "".join(b.text for b in r.content if b.type == "text"), "trace": trace}
        uses = [b for b in r.content if b.type == "tool_use"]
        results = []
        for i, b in enumerate(uses):
            if b.name in NEEDS_APPROVAL:                   # pause: the UI shows Confirm / Cancel
                STATE["pending"] = {"uses": uses[i:], "results": results}
                return {"pending": {"tool": b.name, "cart": cart_snapshot()}, "trace": trace}
            results.append(exec_tool(b, trace))
        STATE["messages"].append({"role": "user", "content": results})
    return {"reply": "Sorry, I couldn't finish that. Let me get a team member.", "trace": trace}

def resume(approved, trace):
    p, STATE["pending"] = STATE["pending"], None
    results = p["results"]
    for b in p["uses"]:
        if b.name in NEEDS_APPROVAL and not approved:
            trace.append({"tool": b.name, "input": b.input, "output": "BLOCKED: customer cancelled"})
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": "The customer did not approve this action."})
        else:
            results.append(exec_tool(b, trace))
    STATE["messages"].append({"role": "user", "content": results})
    return agent_continue(trace)

# ---------- Web server ----------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            return self.send_json({"error": "not found"}, 404)
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "bad request"}, 400)
        with LOCK:
            try:
                trace = []
                if self.path == "/chat":
                    text = str(data.get("message", "")).strip()[:1000]      # validate input
                    if not text:
                        return self.send_json({"error": "Please type a message."}, 400)
                    if STATE["pending"]:
                        return self.send_json({"error": "Please confirm or cancel the order first."}, 409)
                    STATE["messages"].append({"role": "user", "content": text})
                    out = agent_continue(trace)
                elif self.path == "/approve":
                    if not STATE["pending"]:
                        return self.send_json({"error": "Nothing is waiting for approval."}, 409)
                    out = resume(bool(data.get("approved")), trace)
                elif self.path == "/reset":
                    STATE["messages"], STATE["pending"] = [], None
                    out = {"reply": "New conversation started.", "trace": []}
                else:
                    return self.send_json({"error": "not found"}, 404)
                out["cart"] = cart_snapshot()
                self.send_json(out)
            except Exception as e:
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Burger Agent</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--text:#1f1e1b;--muted:#6b6a65;--line:#e4e2dc;--accent:#c2410c;--user:#fde8dc;--ok:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#171614;--panel:#22211e;--text:#ecebe6;--muted:#a3a19a;--line:#38362f;--accent:#fb923c;--user:#3b2a1f;--ok:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--text)}
.wrap{max-width:1100px;margin:0 auto;padding:16px;display:grid;grid-template-columns:1fr 320px;gap:16px;height:100vh}
@media (max-width:820px){.wrap{grid-template-columns:1fr;height:auto}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;display:flex;flex-direction:column;min-height:0}
header{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
header h1{font-size:17px;margin:0;font-weight:600}header span{color:var(--muted);font-size:13px}
#log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px;min-height:300px}
.msg{max-width:85%;padding:10px 13px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.user{align-self:flex-end;background:var(--user)}.bot{align-self:flex-start;border:1px solid var(--line)}
.err{align-self:center;color:var(--bad);font-size:13px}
.approve{align-self:flex-start;border:1px solid var(--accent);border-radius:12px;padding:12px;max-width:85%}
.approve button{margin-right:8px;margin-top:8px}
form{display:flex;gap:8px;padding:12px;border-top:1px solid var(--line)}
input{flex:1;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--text);font:inherit}
button{padding:9px 14px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;cursor:pointer}
button.ghost{background:transparent;color:var(--text);border:1px solid var(--line)}button:disabled{opacity:.5;cursor:default}
.side{padding:14px 16px;overflow-y:auto}.side h2{font-size:14px;margin:4px 0 8px}
.row{display:flex;justify-content:space-between;font-size:14px;padding:2px 0}.total{font-weight:600;border-top:1px solid var(--line);margin-top:6px;padding-top:6px}
.trace{font:12px/1.45 ui-monospace,Menlo,monospace;color:var(--muted);border-left:2px solid var(--line);padding-left:8px;margin-bottom:8px;word-break:break-word}
.trace b{color:var(--text)}.hint{color:var(--muted);font-size:13px}
</style></head><body>
<div class="wrap">
  <div class="card">
    <header><div><h1>Burger Agent</h1><span>Claude + your MCP server, running locally</span></div>
      <button class="ghost" id="reset">New chat</button></header>
    <div id="log"><div class="msg bot">Hi! I can take your order and answer questions about allergens, nutrition, hours, refunds and loyalty. What would you like?</div></div>
    <form id="f"><input id="m" placeholder="e.g. add a classic burger and large fries" autocomplete="off"><button id="send">Send</button></form>
  </div>
  <div class="card side">
    <h2>Cart</h2><div id="cart" class="hint">Empty</div>
    <h2 style="margin-top:18px">MCP tool calls</h2><div id="trace" class="hint">Tool calls will appear here.</div>
  </div>
</div>
<script>
const log=document.getElementById('log'),m=document.getElementById('m'),send=document.getElementById('send');
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function fmt(s){return esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>')}
function add(cls,html){const d=document.createElement('div');d.className=cls;d.innerHTML=html;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
function money(x){return '$'+Number(x).toFixed(2)}
function showCart(c){const el=document.getElementById('cart');
  if(!c||!c.lines){el.className='hint';el.textContent='Empty';return}
  el.className='';el.innerHTML=c.lines.map(l=>`<div class="row"><span>${l.qty} × ${esc(l.item)}${l.size!=='regular'?' ('+esc(l.size)+')':''}</span><span>${money(l.line_total)}</span></div>`).join('')+
  `<div class="row"><span>Tax</span><span>${money(c.tax)}</span></div><div class="row total"><span>Total</span><span>${money(c.total)}</span></div>`}
function showTrace(t){if(!t||!t.length)return;const el=document.getElementById('trace');if(el.classList.contains('hint')){el.className='';el.innerHTML=''}
  t.forEach(x=>{const d=document.createElement('div');d.className='trace';d.innerHTML=`<b>${esc(x.tool)}</b>(${esc(JSON.stringify(x.input))})<br>→ ${esc(x.output.slice(0,160))}`;el.prepend(d)})}
async function post(url,body){send.disabled=true;
  try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
    const d=await r.json();if(d.error){add('err',esc(d.error));return}
    showTrace(d.trace);showCart(d.cart);
    if(d.pending){const c=d.pending.cart,a=add('approve',`<b>Confirm your order</b><br>Total: ${c&&c.total!==undefined?money(c.total):'unknown'}<br><button>Place order</button><button class="ghost">Cancel</button>`);
      const [yes,no]=a.querySelectorAll('button');
      yes.onclick=()=>{yes.disabled=no.disabled=true;post('/approve',{approved:true})};
      no.onclick=()=>{yes.disabled=no.disabled=true;post('/approve',{approved:false})}}
    else add('msg bot',fmt(d.reply||''))}
  catch(e){add('err','Could not reach the app. Is chat_app.py still running?')}
  finally{send.disabled=false;m.focus()}}
document.getElementById('f').onsubmit=e=>{e.preventDefault();const t=m.value.trim();
  if(!t){add('err','Type a message first.');return}m.value='';add('msg user',esc(t));post('/chat',{message:t})};
document.getElementById('reset').onclick=()=>{log.innerHTML='';post('/reset')};
m.focus();
</script></body></html>"""

if __name__ == "__main__":
    print("Burger chat running at http://localhost:8000  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()   # localhost only, not your network
