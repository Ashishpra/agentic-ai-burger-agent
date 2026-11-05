import anthropic
client = anthropic.Anthropic()
r = client.messages.create(model="claude-sonnet-5-5", max_tokens=100,
      messages=[{"role": "user", "content": "Say hello to a future AI architect in one sentence"}])
print(r.content[0].text)
print("Tokens in/out:", r.usage.input_tokens, r.usage.output_tokens)
