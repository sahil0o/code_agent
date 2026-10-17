import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
MODEL = "openai/gpt-oss-120b"

WORKSPACE = Path("playground").resolve()   # the only folder the agent may read


# ---- 1. The tool itself: ordinary Python ----
def read_file(path):
    full_path = (WORKSPACE / path).resolve()
    if not full_path.is_relative_to(WORKSPACE):
        return f"Error: '{path}' is outside the project folder."
    if not full_path.is_file():
        return f"Error: '{path}' not found (or it is a folder)."
    return full_path.read_text()


# ---- 2. The tool's description, written for the model ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the project. "
                           "Use this before answering any question about the code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the project root, e.g. 'inventory.py'.",
                    }
                },
                "required": ["path"],
            },
        },
    }
]

messages = [
    {"role": "system", "content": "You are a coding assistant. Read the relevant files before answering questions about code."},
    {"role": "user", "content": "What does sale_price('bag', 10) return, and is that right?"},
]

# ---- 3. First call: the model asks for a tool ----
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools, max_completion_tokens=2000
)
msg = response.choices[0].message
print("Finish reason:", response.choices[0].finish_reason)

if not msg.tool_calls:
    print("The model answered without using a tool:\n", msg.content)
    raise SystemExit

# ---- 4. Record the request, run the tool, send the result back ----
messages.append({
    "role": "assistant",
    "content": msg.content or "",
    "tool_calls": [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in msg.tool_calls
    ],
})

for tc in msg.tool_calls:
    args = json.loads(tc.function.arguments)   # arguments arrive as a JSON *string*
    print(f">> Model asked for: {tc.function.name}({args})")
    result = read_file(args["path"])
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# ---- 5. Second call: the model answers using the file contents ----
response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools, max_completion_tokens=2000
)
print("Finish reason:", response.choices[0].finish_reason)
print("\nAnswer:\n", response.choices[0].message.content)