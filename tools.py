import json
import difflib
import re
from pathlib import Path
import os

# Workspace: set AGENT_WORKSPACE to point the agent at any project; defaults to ./playground
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", Path(__file__).parent / "playground")).resolve()   # the agent can only see inside this folder
MAX_RESULT_CHARS = 8000                     # cap tool output so one result can't flood the context
MAX_MATCHES = 50
READ_CHUNK_CHARS = 7000                     # how much of a file one read_file call returns

# ---------- Safety helper shared by every tool ----------
def resolve_in_workspace(path):
    full_path = (WORKSPACE / path).resolve()
    if not full_path.is_relative_to(WORKSPACE):
        raise ValueError(f"'{path}' is outside the project folder.")
    return full_path


def is_hidden(path):
    """True for anything under a hidden folder (.git, .pytest_cache) or __pycache__."""
    parts = path.relative_to(WORKSPACE).parts
    return any(part.startswith(".") or part == "__pycache__" for part in parts)


# ---------- The tools ----------
def list_dir(path="."):
    folder = resolve_in_workspace(path)
    if not folder.is_dir():
        raise ValueError(f"'{path}' is not a folder.")
    entries = []
    for item in sorted(folder.iterdir()):
        if is_hidden(item):
            continue
        entries.append(item.name + "/" if item.is_dir() else item.name)
    return "\n".join(entries) or "(empty folder)"


def read_file(path, start_line=1, end_line=None):
    file = resolve_in_workspace(path)
    if not file.is_file():
        raise ValueError(f"'{path}' is not a file. Use list_dir to see which files exist.")

    lines = file.read_text().splitlines()
    total = len(lines)
    if total == 0:
        return "(empty file)"
    start_line = int(start_line)
    if not 1 <= start_line <= total:
        raise ValueError(f"start_line must be between 1 and {total}.")
    last_wanted = total if end_line is None else min(int(end_line), total)

    # Add line numbers, stopping before the chunk gets too big
    output, size, line_no = [], 0, start_line
    while line_no <= last_wanted:
        numbered = f"{line_no:>4} | {lines[line_no - 1]}"
        if output and size + len(numbered) > READ_CHUNK_CHARS:
            break
        output.append(numbered)
        size += len(numbered) + 1
        line_no += 1

    text = "\n".join(output)
    last_shown = line_no - 1
    if last_shown < total:
        text += (f"\n\n[Showing lines {start_line}-{last_shown} of {total}. "
                 f"Call read_file with start_line={last_shown + 1} to read more.]")
    return text


def search(query, path="."):
    root = resolve_in_workspace(path)
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    matches = []
    for file in files:
        if not file.is_file() or is_hidden(file):
            continue
        try:
            lines = file.read_text().splitlines()
        except UnicodeDecodeError:
            continue   # skip binary files
        for number, line in enumerate(lines, start=1):
            if query in line:
                matches.append(f"{file.relative_to(WORKSPACE)}:{number}: {line.strip()}")
                if len(matches) >= MAX_MATCHES:
                    return "\n".join(matches) + f"\n...(stopped at {MAX_MATCHES} matches)"
    return "\n".join(matches) or f"No matches for '{query}'."

def ask_approval(diff_text):
    """Show the proposed change and ask the human. Returns (approved, reason)."""
    print("\n" + "=" * 60)
    print(diff_text)
    print("=" * 60)
    answer = input("Apply this edit? [y = yes / anything else = no, optionally with a reason]: ").strip()
    if answer.lower() in ("y", "yes"):
        return True, ""
    return False, answer

def edit_file(path, old_text, new_text):
    file = resolve_in_workspace(path)
    if not file.is_file():
        raise ValueError(f"'{path}' is not a file. Use list_dir to see which files exist.")
    if file.suffix == ".ipynb":
        raise ValueError("Editing notebooks is not supported yet.")
    if old_text == new_text:
        raise ValueError("old_text and new_text are identical, so there is nothing to change.")

    content = file.read_text()
    count = content.count(old_text)
    if count == 0:
        hint = ""
        if re.search(r"^\s*\d+ \| ", old_text, re.MULTILINE):
            hint = " Remove the line-number prefixes (like '  12 | ') that read_file adds; they are not in the file."
        raise ValueError(f"old_text was not found in '{path}'. It must match the file exactly, "
                         f"including spaces and indentation. Read the file again and copy the text exactly.{hint}")
    if count > 1:
        raise ValueError(f"old_text appears {count} times in '{path}'. Include more surrounding lines "
                         f"so it matches exactly one place.")

    new_content = content.replace(old_text, new_text, 1)
    diff = "".join(difflib.unified_diff(
        content.splitlines(keepends=True), new_content.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))

    approved, reason = ask_approval(diff)
    if not approved:
        return f"The user rejected this edit. Reason: {reason or 'none given'}. Do not retry the same edit."

    file.write_text(new_content)
    return f"Edit applied to '{path}'.\n{diff}"

# ---------- Name -> function, so the loop can look tools up ----------
TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "search": search,
    "edit_file": edit_file,
}



# ---------- The descriptions the model sees ----------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the files and folders inside a project folder. Folder names end with '/'. "
                           "Use this first to see what the project contains.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder path relative to the project root. Use '.' for the root."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the project. Returns numbered lines like '  12 | code'. "
                           "Long files come in parts: the result says which start_line to use for the next part. "
                           "To jump to a specific place, find its line number with search first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the project root, e.g. 'inventory.py'."},
                    "start_line": {"type": "integer", "description": "Line number to start reading from. Defaults to 1."},
                    "end_line": {"type": "integer", "description": "Optional last line to read. Defaults to reading as much as fits."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the project's files for lines containing some text (plain text, case-sensitive). "
                           "Returns matches as 'file:line: text'. Use this to find where a function, class "
                           "or variable is defined or used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The exact text to look for, e.g. 'def sale_price'."},
                    "path": {"type": "string", "description": "Optional folder or file to search in. Defaults to the whole project."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Change a file by replacing one exact piece of text with new text. "
                           "old_text must appear exactly once in the file, copied exactly as it is in the file "
                           "(same spaces and indentation, WITHOUT the line-number prefixes read_file shows). "
                           "Include a few surrounding lines if the text is not unique. "
                           "Always read the file before editing it. The user must approve every edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the project root."},
                    "old_text": {"type": "string", "description": "The exact text to replace."},
                    "new_text": {"type": "string", "description": "The text to put in its place."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


# ---------- One safe entry point: tools never crash the agent ----------
def run_tool(name, arguments_json):
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."
    try:
        args = json.loads(arguments_json or "{}")
        result = TOOLS[name](**args)
    except json.JSONDecodeError:
        return "Error: the tool arguments were not valid JSON."
    except TypeError as e:
        return f"Error: wrong arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {e}"

    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + f"\n...(truncated, {len(result)} characters in total)"
    return result