import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "openai/gpt-oss-120b"

messages = []   # the entire conversation lives here, and only here

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ("exit", "quit"):
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=10,
        messages=messages,   # send the WHOLE history every time
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(f"\nAI: {reply}")
    print(f"[tokens in: {response.usage.prompt_tokens} | out: {response.usage.completion_tokens}]")