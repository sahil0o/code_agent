import json
from pathlib import Path
import os

# Workspace: set AGENT_WORKSPACE to point the agent at any project; defaults to ./playground
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", Path(__file__).parent / "playground")).resolve()   # the agent can only see inside this folder
MAX_RESULT_CHARS = 8000                     # cap tool output so one result can't flood the context
MAX_MATCHES = 50


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


def read_file(path):
    file = resolve_in_workspace(path)
    if not file.is_file():
        raise ValueError(f"'{path}' is not a file. Use list_dir to see which files exist.")
    return file.read_text()


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


# ---------- Name -> function, so the loop can look tools up ----------
TOOLS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "search": search,
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
            "description": "Read the full contents of a file in the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the project root, e.g. 'inventory.py'."}
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