import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # point the OpenAI SDK at Groq
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "openai/gpt-oss-120b"

response = client.chat.completions.create(
    model=MODEL,
    max_completion_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain what a coding agent is in two sentences."}
    ],
)

print(response)   # print the raw object once, just to see its shape
print("-" * 50)

choice = response.choices[0]
print(choice.message.content)

print("-" * 50)
print("Finish reason:", choice.finish_reason)
print("Tokens:", response.usage.prompt_tokens, "in /", response.usage.completion_tokens, "out")