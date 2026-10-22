"""The agent's tools. Everything a tool needs (which folder, how to ask for approval) lives on a
Workspace object, so nothing here is a global that other files have to overwrite."""
import difflib
import inspect
import json
import re
import shlex
import sys
from pathlib import Path

from runner import LocalRunner

MAX_RESULT_CHARS = 8000      # cap tool output so one result can't flood the context
MAX_MATCHES = 50
READ_CHUNK_CHARS = 7000      # how much of a file one read_file call returns
MAX_NEW_FILE_CHARS = 100_000 # create_file refuses anything bigger (also keeps approval diffs small enough for the relay)

# run_command: which programs may run, and what they really mean. "python" is the SAME Python that runs
# the agent (your virtual environment), which also works on Windows where "python3" often doesn't exist.
ALLOWED_PROGRAMS = {
    "python": [sys.executable],
    "python3": [sys.executable],
    "pytest": [sys.executable, "-m", "pytest"],
}
SHELL_SYMBOLS = ("&&", "||", "|", ";", ">", "<", "`", "$(")   # meaningless without a shell, so refuse clearly
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300
MAX_OUTPUT_CHARS = 3500      # per stream (stdout / stderr); we keep the END, where errors and summaries are


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
    answer = input("Approve? [y = yes / anything else = no, optionally with a reason]: ").strip()
    if answer.lower() in ("y", "yes"):
        return True, ""
    return False, answer


def auto_approval(diff_text):
    """Approves everything. Only for evals, which run on throwaway copies."""
    return True, ""


# ---------- One project folder plus the tools that work inside it ----------
class Workspace:
    def __init__(self, root, approve=terminal_approval, runner=None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace folder not found: {self.root}")
        self.approve = approve          # any function: text to approve -> (approved, reason)
        self.runner = runner or LocalRunner()   # HOW commands run (later: a Docker sandbox)
        self.tools = {                  # name the model uses -> the method that runs
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "search": self.search,
            "edit_file": self.edit_file,
            "create_file": self.create_file,
            "run_command": self.run_command,
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

    def create_file(self, path, content):
        file = self.resolve(path)
        if file.is_dir() or path.endswith(("/", "\\")):
            raise ValueError(f"'{path}' is a folder. Give a file name, e.g. 'scripts/plot.py'.")
        if file.exists():
            raise ValueError(f"'{path}' already exists. Use edit_file to change an existing file.")
        if self.is_hidden(file):
            raise ValueError(f"'{path}' is inside a hidden folder (like .git or .venv). Creating files there is not allowed.")
        if len(content) > MAX_NEW_FILE_CHARS:
            raise ValueError(f"content is {len(content)} characters; the limit is {MAX_NEW_FILE_CHARS}. "
                             f"Create a smaller file.")

        # Show the whole new file as a diff against nothing, so the approval looks like any other edit
        lines = content.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        diff = "".join(difflib.unified_diff([], lines, fromfile="/dev/null", tofile=f"b/{path}"))

        approved, reason = self.approve(diff)
        if not approved:
            return f"The user rejected creating '{path}'. Reason: {reason or 'none given'}. Do not retry the same file."
        file.parent.mkdir(parents=True, exist_ok=True)          # create missing folders, e.g. scripts/
        with open(file, "w", encoding="utf-8", newline="") as f: # UTF-8 on every OS; keep \n line endings as written
            f.write(content)
        return f"Created '{path}' ({len(lines)} lines)."

    def run_command(self, command, timeout=DEFAULT_TIMEOUT):
        # 1. Refuse shell tricks: there is no shell, so "a && b" or "x | y" would not do what the model expects
        used = [s for s in SHELL_SYMBOLS if s in command]
        if used:
            raise ValueError(f"Shell features are not supported ({' '.join(used)}). Run one command at a time, "
                             f"e.g. 'python plot.py'.")
        # 2. Split into program + arguments, the way a terminal would (quotes keep spaces together)
        try:
            args = shlex.split(command)
        except ValueError as e:
            raise ValueError(f"Could not read the command: {e}. Check the quotes.")
        if not args:
            raise ValueError("The command is empty.")
        program = args[0].lower().removesuffix(".exe")
        if program not in ALLOWED_PROGRAMS:
            raise ValueError(f"'{args[0]}' is not allowed. Allowed programs: {', '.join(ALLOWED_PROGRAMS)}.")
        timeout = max(1, min(int(timeout), MAX_TIMEOUT))

        # 3. Ask the human, showing exactly what will run and where
        approved, reason = self.approve(f"Run in {self.root.name} (timeout {timeout}s):\n$ {command}\n")
        if not approved:
            return f"The user rejected running this command. Reason: {reason or 'none given'}. Do not retry the same command."

        # 4. Run it (inside the workspace) and report back what happened
        result = self.runner.run(ALLOWED_PROGRAMS[program] + args[1:], cwd=self.root, timeout=timeout)
        if result.timed_out:
            status = f"Timed out after {timeout}s, so the command was stopped."
        else:
            status = f"Exit code: {result.exit_code}" + (" (success)" if result.exit_code == 0 else " (failed)")
        return f"{status}\n--- stdout ---\n{tail(result.stdout)}\n--- stderr ---\n{tail(result.stderr)}"

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


def tail(text):
    """Keep the end of long output: that is where tracebacks and test summaries are."""
    text = text.strip() or "(empty)"
    if len(text) > MAX_OUTPUT_CHARS:
        text = f"...({len(text) - MAX_OUTPUT_CHARS} earlier characters cut)\n" + text[-MAX_OUTPUT_CHARS:]
    return text


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
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a NEW file with the given content. Fails if the file already exists "
                           "(use edit_file for existing files). Missing folders are created. "
                           "The user must approve every new file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "New file path relative to the project root, e.g. 'scripts/plot.py'."},
                    "content": {"type": "string", "description": "The full text of the new file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run ONE command in the project folder and get its exit code and output. "
                           "Allowed programs: python, pytest (e.g. 'python plot.py', 'pytest -q', "
                           "'pytest tests/test_x.py'). No shell features: no &&, |, >, ; . "
                           "Use forward slashes in paths. The user must approve every command. "
                           "Use it to check your work: run the script or the tests, read the error, fix, run again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command, e.g. 'python plot.py'."},
                    "timeout": {"type": "integer", "description": f"Seconds before the command is stopped. "
                                                                  f"Default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT}."},
                },
                "required": ["command"],
            },
        },
    },
]