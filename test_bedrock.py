import sys
from anthropic import AnthropicBedrock

model_id = sys.argv[1]
client = AnthropicBedrock(aws_region="us-east-1")   # uses ~/.aws credentials: bedrock-course

r = client.messages.create(
    model=model_id, max_tokens=100,
    messages=[{"role": "user", "content": "Say hello to a future AI architect, from AWS Bedrock, in one sentence"}])
print("".join(b.text for b in r.content if b.type == "text"))
print("Tokens in/out:", r.usage.input_tokens, r.usage.output_tokens)
