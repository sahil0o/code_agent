"""The agent's tools. Everything a tool needs (which folder, how to ask for approval) lives on a
Workspace object, so nothing here is a global that other files have to overwrite."""
import difflib
import inspect
import json
import re
from pathlib import Path

MAX_RESULT_CHARS = 8000      # cap tool output so one result can't flood the context
MAX_MATCHES = 50
READ_CHUNK_CHARS = 7000      # how much of a file one read_file call returns


# ---------- Helpers that don't depend on any workspace ----------
def notebook_to_text(file):
    """Turn a .ipynb file (JSON) into readable text: just the cells, no outputs or metadata."""
    try:
        notebook = json.loads(file.read_text())
    except json.JSONDecodeError:
        raise ValueError(f"'{file.name}' is not a valid notebook (its JSON is broken).")
    parts = []
    for number, cell in enumerate(notebook.get("cells", []), start=1):
        source = "".join(cell.get("source", []))
        parts.append(f"# ---- cell {number} ({cell.get('cell_type')}) ----\n{source}")
    return "\n\n".join(parts)


def read_for_agent(file):
    """How the agent sees a file. Notebooks get cleaned up; everything else is plain text."""
    if file.suffix == ".ipynb":
        return notebook_to_text(file)
    return file.read_text()


def terminal_approval(diff_text):
    """The default approver: show the diff in the terminal and ask. Returns (approved, reason)."""
    print("\n" + "=" * 60)
    print(diff_text)
    print("=" * 60)
    answer = input("Apply this edit? [y = yes / anything else = no, optionally with a reason]: ").strip()
    if answer.lower() in ("y", "yes"):
        return True, ""
    return False, answer


def auto_approval(diff_text):
    """Approves everything. Only for evals, which run on throwaway copies."""
    return True, ""


# ---------- One project folder plus the tools that work inside it ----------
class Workspace:
    def __init__(self, root, approve=terminal_approval):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace folder not found: {self.root}")
        self.approve = approve          # any function: diff_text -> (approved, reason)
        self.tools = {                  # name the model uses -> the method that runs
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "search": self.search,
            "edit_file": self.edit_file,
        }

    # ----- safety helpers -----
    def resolve(self, path):
        full_path = (self.root / path).resolve()
        if not full_path.is_relative_to(self.root):
            raise ValueError(f"'{path}' is outside the project folder.")
        return full_path

    def is_hidden(self, path):
        """True for anything under a hidden folder (.git, .venv) or __pycache__."""
        parts = path.relative_to(self.root).parts
        return any(part.startswith(".") or part == "__pycache__" for part in parts)

    # ----- the tools -----
    def list_dir(self, path="."):
        folder = self.resolve(path)
        if not folder.is_dir():
            raise ValueError(f"'{path}' is not a folder.")
        entries = [item.name + "/" if item.is_dir() else item.name
                   for item in sorted(folder.iterdir()) if not self.is_hidden(item)]
        return "\n".join(entries) or "(empty folder)"

    def read_file(self, path, start_line=1, end_line=None, line_start=None, line_end=None):
        start_line = line_start or start_line      # the model sometimes uses these names
        end_line = line_end or end_line
        file = self.resolve(path)
        if not file.is_file():
            raise ValueError(f"'{path}' is not a file. Use list_dir to see which files exist.")

        lines = read_for_agent(file).splitlines()
        total = len(lines)
        if total == 0:
            return "(empty file)"
        start_line = int(start_line)
        if not 1 <= start_line <= total:
            raise ValueError(f"start_line must be between 1 and {total}.")

        last_wanted = total if end_line is None else min(int(end_line), total)
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

    def search(self, query, path="."):
        root = self.resolve(path)
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        matches = []
        for file in files:
            if not file.is_file() or self.is_hidden(file):
                continue
            try:
                lines = read_for_agent(file).splitlines()
            except (UnicodeDecodeError, ValueError):
                continue   # skip binary files and broken notebooks
            for number, line in enumerate(lines, start=1):
                if query.lower() in line.lower():
                    matches.append(f"{file.relative_to(self.root)}:{number}: {line.strip()}")
                    if len(matches) >= MAX_MATCHES:
                        return "\n".join(matches) + f"\n...(stopped at {MAX_MATCHES} matches)"
        return "\n".join(matches) or f"No matches for '{query}'."

    def edit_file(self, path, old_text, new_text):
        file = self.resolve(path)
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

        approved, reason = self.approve(diff)
        if not approved:
            return f"The user rejected this edit. Reason: {reason or 'none given'}. Do not retry the same edit."
        file.write_text(new_content)
        return f"Edit applied to '{path}'.\n{diff}"

    # ----- one safe entry point: tools never crash the agent -----
    def run(self, name, arguments_json):
        if name not in self.tools:
            return f"Error: unknown tool '{name}'. Available tools: {', '.join(self.tools)}."
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return "Error: the tool arguments were not valid JSON."
        try:
            result = self.tools[name](**args)
        except TypeError:
            params = inspect.signature(self.tools[name]).parameters
            required = [p for p, v in params.items() if v.default is inspect.Parameter.empty]
            return (f"Error: wrong arguments for {name}. Allowed: {', '.join(params)}. "
                    f"Required: {', '.join(required)}.")
        except Exception as e:
            return f"Error: {e}"

        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + f"\n...(truncated, {len(result)} characters in total)"
        return result


# ---------- The descriptions the model sees (the same for every workspace) ----------
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
                    "end_line": {"type": "integer", "description": "Optional last line to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the project's files for lines containing some text (plain text, case-insensitive). "
                           "Returns matches as 'file:line: text'. Use this to find where a function, class "
                           "or variable is defined or used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The text to look for, e.g. 'def sale_price'."},
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