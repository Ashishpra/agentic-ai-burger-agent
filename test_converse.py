import sys, boto3

model_id = sys.argv[1]
client = boto3.client("bedrock-runtime", region_name="us-east-1")   # uses bedrock-course credentials

r = client.converse(
    modelId=model_id,
    messages=[{"role": "user", "content": [{"text": "Say hello to a future AI architect, from AWS Bedrock, in one sentence"}]}],
    inferenceConfig={"maxTokens": 100},
)
print(r["output"]["message"]["content"][0]["text"])
print("Tokens in/out:", r["usage"]["inputTokens"], r["usage"]["outputTokens"])
print("Latency (ms):", r["metrics"]["latencyMs"])
