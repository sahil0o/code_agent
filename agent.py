import os
import sys
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv

from tools import TOOL_SCHEMAS, TOOLS, run_tool

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=5,   # on a rate-limit error, wait and retry automatically
)
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 10      # safety cap: the agent can never loop forever

SYSTEM_PROMPT = (
    "You are a coding assistant working on a Python project. "
    "Explore the project with your tools before answering: list_dir to see files, "
    "search to find where things are defined or used, read_file to read code. "
    "Never guess file names or code you have not read."
)


def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        # 1. Ask the model what to do next
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                max_completion_tokens=4000,
            )
        except BadRequestError as e:
            # Some 400 errors mean the MODEL produced malformed output. Retrying usually works.
            error_text = str(e)
            if "tool_use_failed" in error_text:
                note = f"Your last tool call was invalid. Only these tools exist: {', '.join(TOOLS)}. Try again."
            elif "output_parse_failed" in error_text:
                note = "Your last response could not be read. Reply with either a tool call or your final answer."
            else:
                raise   # a different problem: crash loudly so we notice

            body = e.body if isinstance(e.body, dict) else {}            # NEW
            body = body.get("error", body)                               # NEW
            failed = str(body.get("failed_generation", "(not provided)")) # NEW

            print(f"[step {step}] the API rejected the model's output, asking it to try again")
            print(f"           ✗ {failed[:200]}")                        # NEW
            messages.append({"role": "user", "content": note})
            continue

        choice = response.choices[0]
        msg = choice.message

        # 2. No tool calls means the model is finished
        if not msg.tool_calls:
            return msg.content or f"(empty answer, finish_reason={choice.finish_reason})"

        # 3. Record what the model asked for
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

        # 4. Run every requested tool and send back the results
        for tc in msg.tool_calls:
            print(f"[step {step}] {tc.function.name}({tc.function.arguments})")
            result = run_tool(tc.function.name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return f"Stopped after {MAX_STEPS} steps without a final answer."


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What does sale_price('bag', 10) return, and is that right?"
    print("\nAnswer:\n" + run_agent(question))