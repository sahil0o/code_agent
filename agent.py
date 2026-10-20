import os
import sys
import time
from openai import OpenAI, BadRequestError
from dotenv import load_dotenv
import tools

from tools import TOOL_SCHEMAS, TOOLS, run_tool
from logger import RunLogger

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=5,   # on a rate-limit error, wait and retry automatically
)
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 15      # safety cap: the agent can never loop forever

SYSTEM_PROMPT = """You are a coding assistant that answers questions about the project in your workspace.

How to work:
1. Explore before answering. Use list_dir to see the files, search to find where things are defined or used, and read_file to read the code.
2. Read the actual code behind every claim. Docstrings, comments and search snippets are hints, not proof.
3. Follow names through the code. If a value is stored in a variable or constant (for example PATH = "out.csv"), search for that name to find where it is used.
4. Long files come in parts. If read_file says more lines exist, read them whenever they might matter.
5. Do not repeat a search that already returned results. Try a different query or read the file instead.
6. Never guess file names, function names or behaviour you have not read.
7. If a tool keeps returning the same result or an error you cannot fix, stop and tell the user what you found instead of retrying.

How to edit:
- Read the file before editing it, and copy old_text exactly from what you read, without the line-number prefixes.
- Make small, focused edits: change only what the task needs.
- If the user rejects an edit, follow their reason instead of retrying the same change.

How to answer:
- Answer the question first, then give the evidence.
- Back every factual claim with a citation like file.py:42 or file.py:40-45, using the line numbers read_file shows.
- Mention edge cases you saw in the code, such as empty inputs, missing files or error branches.
- If you could not verify something, say so plainly instead of guessing."""


def call_model(messages, log, step, **extra):
    """One API call, timed and logged."""
    started = time.time()
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMAS, max_completion_tokens=4000, **extra
    )
    log.model_call(step, response.usage, time.time() - started, response.choices[0].finish_reason)
    return response


def run_agent(question, log=None):
    log = log or RunLogger(question, MODEL, tools.WORKSPACE)
    print(f"Workspace: {tools.WORKSPACE}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        for step in range(1, MAX_STEPS + 1):
            # 1. Ask the model what to do next
            try:
                response = call_model(messages, log, step)
            except BadRequestError as e:
                # Some 400 errors mean the MODEL produced malformed output. Retrying usually works.
                error_text = str(e)
                if "tool_use_failed" in error_text:
                    code = "tool_use_failed"
                    note = f"Your last tool call was invalid. Only these tools exist: {', '.join(TOOLS)}. Try again."
                elif "output_parse_failed" in error_text:
                    code = "output_parse_failed"
                    note = "Your last response could not be read. Reply with either a tool call or your final answer."
                else:
                    raise   # a different problem: crash loudly so we notice

                body = e.body if isinstance(e.body, dict) else {}
                body = body.get("error", body)
                failed = str(body.get("failed_generation", "(not provided)"))
                log.api_error(step, code, failed)

                print(f"[step {step}] the API rejected the model's output, asking it to try again")
                print(f"           ✗ {failed[:200]}")
                messages.append({"role": "user", "content": note})
                continue

            msg = response.choices[0].message

            # 2. No tool calls means the model is finished
            if not msg.tool_calls:
                answer = msg.content or f"(empty answer, finish_reason={response.choices[0].finish_reason})"
                return finish(log, "answered", answer)

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
                preview = result.replace("\n", " | ")[:100]
                print(f"           → {preview}")
                log.tool_call(step, tc.function.name, tc.function.arguments, result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Out of steps: ask for a best-effort answer from what it has already gathered
        print(f"[limit] {MAX_STEPS} steps used, asking for a final answer")
        messages.append({"role": "user", "content": "You have run out of steps. Answer now using only what you have "
                         "already found, with citations, and say clearly what you could not verify."})
        try:
            response = call_model(messages, log, MAX_STEPS + 1, tool_choice="none")
            answer = response.choices[0].message.content or f"Stopped after {MAX_STEPS} steps without a final answer."
        except BadRequestError:
            answer = f"Stopped after {MAX_STEPS} steps without a final answer."
        return finish(log, "step_limit", answer)

    except Exception as e:
        finish(log, "crashed", f"{type(e).__name__}: {e}")
        raise   # still crash loudly, but the log is saved first


def finish(log, outcome, answer):
    path = log.finish(outcome, answer)
    s = log.record["summary"]
    print(f"\n[run] {s['outcome']} | {s['steps']} steps | {s['tool_calls']} tool calls "
          f"({s['tool_errors']} errors) | {s['tokens_in']} tokens in, {s['tokens_out']} out "
          f"| {s['seconds']}s | log: {path.name}")
    return answer


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What does sale_price('bag', 10) return, and is that right?"
    print("\nAnswer:\n" + run_agent(question))