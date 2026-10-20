import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm import LLM, BadModelOutput
from logger import RunLogger
from tools import TOOL_SCHEMAS, Workspace

load_dotenv()
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


def call_model(llm, messages, log, step, **extra):
    """One model call, logged."""
    response, seconds = llm.chat(messages, tools=TOOL_SCHEMAS, **extra)
    log.model_call(step, response.usage, seconds, response.choices[0].finish_reason)
    return response


def run_agent(question, workspace, llm, log=None):
    log = log or RunLogger(question, llm.model, workspace.root)
    print(f"Workspace: {workspace.root} | Model: {llm.model}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        for step in range(1, MAX_STEPS + 1):
            # 1. Ask the model what to do next
            try:
                response = call_model(llm, messages, log, step)
            except BadModelOutput as e:
                # The model produced malformed output. Tell it what went wrong and let it try again.
                if e.code == "tool_use_failed":
                    note = f"Your last tool call was invalid. Only these tools exist: {', '.join(workspace.tools)}. Try again."
                else:
                    note = "Your last response could not be read. Reply with either a tool call or your final answer."
                log.api_error(step, e.code, e.failed_text)
                print(f"[step {step}] the API rejected the model's output, asking it to try again")
                print(f"           ✗ {e.failed_text[:200]}")
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
                result = workspace.run(tc.function.name, tc.function.arguments)
                preview = result.replace("\n", " | ")[:100]
                print(f"           → {preview}")
                log.tool_call(step, tc.function.name, tc.function.arguments, result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Out of steps: ask for a best-effort answer from what it has already gathered
        print(f"[limit] {MAX_STEPS} steps used, asking for a final answer")
        messages.append({"role": "user", "content": "You have run out of steps. Answer now using only what you have "
                         "already found, with citations, and say clearly what you could not verify."})
        try:
            response = call_model(llm, messages, log, MAX_STEPS + 1, tool_choice="none")
            answer = response.choices[0].message.content or f"Stopped after {MAX_STEPS} steps without a final answer."
        except BadModelOutput:
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
    workspace = Workspace(os.environ.get("AGENT_WORKSPACE", Path(__file__).parent / "playground"))
    llm = LLM(
        provider=os.environ.get("AGENT_PROVIDER", "groq"),
        model=os.environ.get("AGENT_MODEL"),          # empty means the provider's default model
        base_url=os.environ.get("AGENT_BASE_URL"),    # only for servers not in llm.PROVIDERS
    )
    print("\nAnswer:\n" + run_agent(question, workspace, llm))