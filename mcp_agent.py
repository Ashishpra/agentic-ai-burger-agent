"""MCP host: an agent that DISCOVERS its tools from an MCP server at runtime, instead of hard-coding them."""
import asyncio, json
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(command="python3", args=["burger_mcp_server.py"])
MODEL = "claude-haiku-5-5"
NEEDS_APPROVAL = {"checkout"}     # the HOST asks the human before these tools run
SYSTEM = ("You are a friendly ordering assistant for a burger restaurant. Use the tools for prices, "
          "the cart and store information; never guess or do math yourself. Keep replies short.")

client = anthropic.Anthropic()

async def main():
    async with stdio_client(SERVER) as (read, write):          # starts the server as a child process
        async with ClientSession(read, write) as session:
            await session.initialize()                          # MCP handshake

            # ---------- DISCOVERY: ask the server what it offers ----------
            listed = await session.list_tools()
            tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                     for t in listed.tools]
            print("Discovered tools:", [t["name"] for t in tools])
            resources = await session.list_resources()
            print("Discovered resources:", [str(r.uri) for r in resources.resources])
            print("Type 'quit' to exit.")

            messages = []
            while True:
                user = (await asyncio.to_thread(input, "\nYOU: ")).strip()
                if user.lower() in ("quit", "exit"):
                    break
                if not user:
                    continue
                messages.append({"role": "user", "content": user})

                for _ in range(6):                                  # the same ReAct loop as Day 2
                    r = client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM,
                                               tools=tools, messages=messages)
                    messages.append({"role": "assistant", "content": r.content})
                    if r.stop_reason != "tool_use":
                        print("AGENT:", "".join(b.text for b in r.content if b.type == "text"))
                        break
                    results = []
                    for b in r.content:
                        if b.type != "tool_use":
                            continue
                        if b.name in NEEDS_APPROVAL:
                            ans = await asyncio.to_thread(input, f"   >>> [HOST APPROVAL] run '{b.name}'? (y/n): ")
                            if ans.strip().lower() not in ("y", "yes"):
                                results.append({"type": "tool_result", "tool_use_id": b.id,
                                                "content": "The customer did not approve this action."})
                                print(f"   [host] blocked {b.name}")
                                continue
                        res = await session.call_tool(b.name, b.input)   # the call goes over MCP
                        text = "".join(c.text for c in res.content if c.type == "text")
                        print(f"   [mcp] {b.name}({json.dumps(b.input)}) -> {text[:140]}")
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": text})
                    messages.append({"role": "user", "content": results})

asyncio.run(main())
