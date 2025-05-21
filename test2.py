# import boto3
# import json 

# client = boto3.client(service_name="bedrock-runtime")

# prompt = "write a short poem"

# messages = [
#     {"role": "user", "content": [{"text": prompt}]},
# ]

# model_response = client.converse(
#     modelId="us.amazon.nova-micro-v1:0", 
#     messages=messages
# )

# print("\\n[Full Response]")
# print(json.dumps(model_response, indent=2))

# print("\\n[Response Content Text]")
# print(model_response["output"]["message"]["content"][0]["text"])

import boto3
import json

client = boto3.client(service_name="bedrock-runtime")


system = "You are a helpful assistant that writes creative and concise poems."
prompt = "Write a haiku of 4 lines."

# Combine system + user instruction
combined_prompt = f"{system}\n\n{prompt}"

messages = [
    {"role": "user", "content": [{"text": prompt}]},
    {"role": "assistant", "content": [{"text": "You are a writer that likes to write about mountains."}]}
]

model_response = client.converse(
    modelId="us.amazon.nova-micro-v1:0", 
    messages=messages,
    inferenceConfig={
        "temperature": 0.0  # <-- deterministic output
    }
)

print("\n[Full Response]")
print(json.dumps(model_response, indent=2))

print("\n[Response Content Text]")
print(model_response["output"]["message"]["content"][0]["text"])