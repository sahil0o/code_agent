"""Worker: runs on your Mac. Connects the agent to the relay, so a phone can control it.

Run the relay first, then:  python worker.py
Press Enter in this terminal at any time to get a fresh pairing link (e.g. for another device).
"""
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx
import qrcode
from dotenv import load_dotenv

from agent import Agent
from llm import LLM
from logger import RunLogger
from tools import Workspace

load_dotenv()     # so RELAY_WORKER_KEY, AGENT_WORKSPACES etc. can live in .env

RELAY_URL = os.environ.get("RELAY_URL", "http://127.0.0.1:8000").rstrip("/")
WORKER_KEY = os.environ.get("RELAY_WORKER_KEY", "")
APPROVAL_TIMEOUT = 600      # seconds to wait for the phone's answer before treating it as "no"
MAX_DIFF_CHARS = 100_000    # keep approval requests under the relay's size limit


class RelayConnection:
    """Everything the worker says to, or asks of, the relay."""

    def __init__(self, base_url):
        self.http = httpx.Client(base_url=base_url, timeout=40,
                                 verify=os.environ.get("RELAY_CA") or True)
        self.session_id = None

    def register(self):
        """Create a new session. Also used to recover after the relay restarts."""
        response = self.http.post("/sessions", headers={"X-Worker-Key": WORKER_KEY})
        if response.status_code == 401:
            raise SystemExit("The relay rejected the worker key. Check RELAY_WORKER_KEY matches on both sides.")
        response.raise_for_status()
        info = response.json()
        self.session_id = info["session_id"]
        self.http.headers["Authorization"] = info["worker_token"]
        show_pairing_link(self.session_id, info["pairing_code"], info["pairing_seconds"])

    def new_pairing(self):
        info = self.http.post(f"/sessions/{self.session_id}/pairing").json()
        show_pairing_link(self.session_id, info["pairing_code"], info["pairing_seconds"])

    def send(self, event_type, **data):
        try:
            self.http.post(f"/sessions/{self.session_id}/events", json={"type": event_type, **data})
        except httpx.HTTPError as e:
            print(f"[worker] could not send '{event_type}' event: {e}")

    def next_message(self):
        """Long poll: returns the phone's next message, or None after ~25 s of silence."""
        response = self.http.get(f"/sessions/{self.session_id}/inbox")
        response.raise_for_status()
        return response.json()["message"]


def show_pairing_link(session_id, code, seconds):
    link = f"{RELAY_URL}/#session={session_id}&pair={code}"
    qr = qrcode.QRCode(border=1)
    qr.add_data(link)
    qr.print_ascii(invert=True)
    print(f"[worker] scan the QR code, or open this link on your phone:\n         {link}")
    print(f"[worker] the link works ONCE and expires in {seconds // 60} minutes. "
          f"Press Enter here for a new one.\n")


class RelayLogger(RunLogger):
    """A normal RunLogger that also streams every event to the phone."""

    def __init__(self, relay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relay = relay

    def _add(self, event_type, **data):
        super()._add(event_type, **data)
        self.relay.send(event_type, **data)


class RemoteApprovals:
    """Replaces the keyboard prompt: sends the diff to the phone and waits for the answer."""

    def __init__(self, relay):
        self.relay = relay
        self.pending = {}
        self.lock = threading.Lock()

    def ask(self, diff_text):
        approval_id = uuid.uuid4().hex[:8]
        waiter = {"done": threading.Event(), "answer": None}
        with self.lock:
            self.pending[approval_id] = waiter
        if len(diff_text) > MAX_DIFF_CHARS:
            diff_text = diff_text[:MAX_DIFF_CHARS] + "\n...(diff too long to show in full)"
        kind = "command" if diff_text.startswith("Run in ") else "edit"   # so the phone can show the right heading
        self.relay.send("approval_needed", id=approval_id, diff=diff_text, kind=kind)
        print(f"[worker] waiting for approval {approval_id} from the phone...")

        answered = waiter["done"].wait(timeout=APPROVAL_TIMEOUT)
        with self.lock:
            self.pending.pop(approval_id, None)
        if not answered:
            self.relay.send("approval_resolved", id=approval_id, approved=False)
            return False, "no answer from the phone in time"
        answer = waiter["answer"]
        approved = bool(answer.get("approved"))
        self.relay.send("approval_resolved", id=approval_id, approved=approved)
        return approved, answer.get("reason", "")

    def deliver(self, message):
        with self.lock:
            waiter = self.pending.get(message.get("id"))
        if waiter is None:
            self.relay.send("notice", text=f"No edit is waiting for approval id {message.get('id')}.")
            return
        waiter["answer"] = message
        waiter["done"].set()


def load_workspaces(start_folder):
    """Folders the phone may switch between, set ON THE MAC in .env, e.g.
    AGENT_WORKSPACES=practice=~/code/playground,textkit=~/code/textkit
    The phone only ever sends a NAME from this list, never a path."""
    start_folder = Path(start_folder).expanduser().resolve()
    spaces = {start_folder.name: start_folder}          # the starting workspace is always allowed
    for item in os.environ.get("AGENT_WORKSPACES", "").split(","):
        if "=" not in item:
            continue
        name, path = (part.strip() for part in item.split("=", 1))
        folder = Path(path).expanduser().resolve()
        if folder.is_dir():
            spaces[name] = folder
        else:
            print(f"[worker] skipping workspace '{name}': {folder} is not a folder")
    return spaces


def load_models(default_llm):
    """Models the phone may switch between, set ON THE MAC in .env, e.g.
    AGENT_MODELS=gptoss=groq:openai/gpt-oss-120b,small=groq:openai/gpt-oss-20b,local=ollama:qwen2.5-coder:7b
    Each entry is  label=provider:model . The phone only ever sends a LABEL from this list;
    API keys stay on the Mac."""
    models = {default_llm.model: default_llm}      # the starting model is always allowed
    for item in os.environ.get("AGENT_MODELS", "").split(","):
        if "=" not in item:
            continue
        label, spec = (part.strip() for part in item.split("=", 1))
        provider, _, model = spec.partition(":")   # split at the FIRST colon: "ollama:qwen2.5-coder:7b" works
        try:
            models[label] = LLM(provider=provider.strip(), model=model.strip() or None)
        except ValueError as e:                     # unknown provider, missing API key, ...
            print(f"[worker] skipping model '{label}': {e}")
    return models


class Worker:
    """What the Mac is doing right now: which model, which folders are allowed, which one is active."""

    def __init__(self, relay, models, current_model, workspaces, current):
        self.relay = relay
        self.models = models                  # label -> LLM (the model allowlist)
        self.current_model = current_model    # label of the active model
        self.workspaces = workspaces          # name -> folder (the folder allowlist)
        self.current = current                # name of the active folder
        self.approvals = RemoteApprovals(relay)
        self.task = None                      # the thread running the current task, if any

    def busy(self):
        return bool(self.task and self.task.is_alive())

    @property
    def llm(self):
        return self.models[self.current_model]

    def announce(self):
        self.relay.send("worker_online", workspace=self.current, workspaces=sorted(self.workspaces),
                        model=self.current_model, models=sorted(self.models))

    def switch_model(self, label):
        if self.busy():
            self.relay.send("notice", text="Can't switch model while a task is running.")
        elif label not in self.models:
            self.relay.send("notice", text=f"Unknown model '{label}'. Allowed: {', '.join(sorted(self.models))}")
        else:
            self.current_model = label
            print(f"[worker] model switched to {label}: {self.llm}")
        self.relay.send("model_changed", model=self.current_model)   # tell the page what is active

    def switch_workspace(self, name):
        if self.busy():
            self.relay.send("notice", text="Can't switch workspace while a task is running.")
        elif name not in self.workspaces:
            self.relay.send("notice", text=f"Unknown workspace '{name}'. Allowed: {', '.join(sorted(self.workspaces))}")
        else:
            self.current = name
            print(f"[worker] workspace switched to {name}: {self.workspaces[name]}")
        self.relay.send("workspace_changed", workspace=self.current)   # tell the page what is active

    def start_task(self, text):
        if self.busy():
            self.relay.send("notice", text="Busy with another task. Try again when it finishes.")
            return
        print(f"[worker] new task: {text}")
        self.task = threading.Thread(target=self.run_task, args=(text,), daemon=True)
        self.task.start()

    def run_task(self, text):
        """Runs in a background thread, so the main loop keeps receiving approvals meanwhile."""
        self.relay.send("task_started", text=text)
        try:
            workspace = Workspace(self.workspaces[self.current], approve=self.approvals.ask)
            log = RelayLogger(self.relay, text, self.llm.model, workspace.root)
            answer = Agent(self.llm, workspace).run(text, log=log)
            self.relay.send("answer", text=answer, summary=log.record.get("summary", {}))
        except Exception as e:
            self.relay.send("error", text=f"{type(e).__name__}: {e}")


def watch_keyboard(relay):
    """Press Enter in the worker's terminal to print a new pairing link."""
    for _ in sys.stdin:
        try:
            relay.new_pairing()
        except httpx.HTTPError as e:
            print(f"[worker] could not get a new pairing link: {e}")


def main():
    if len(WORKER_KEY) < 32:
        raise SystemExit("Set RELAY_WORKER_KEY (the same secret the relay uses) in .env or the environment.")
    default_llm = LLM.from_env()
    models = load_models(default_llm)
    workspaces = load_workspaces(os.environ.get("AGENT_WORKSPACE", Path(__file__).parent / "playground"))
    current = next(iter(workspaces))          # the starting folder is first in the list
    relay = RelayConnection(RELAY_URL)
    worker = Worker(relay, models, default_llm.model, workspaces, current)

    print(f"[worker] relay: {RELAY_URL}")
    print(f"[worker] models the phone may choose: {', '.join(sorted(models))}")
    print(f"[worker] current model: {default_llm}")
    print(f"[worker] workspaces the phone may choose: {', '.join(sorted(workspaces))}")
    print(f"[worker] current workspace: {workspaces[current]}\n")
    relay.register()
    worker.announce()
    if sys.stdin.isatty():
        threading.Thread(target=watch_keyboard, args=(relay,), daemon=True).start()

    failures = 0
    while True:
        try:
            message = relay.next_message()
            failures = 0
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # The relay forgot our session (it restarted, or the session expired). Start a new one.
                print("[worker] the relay no longer knows this session (it probably restarted). Creating a new one.")
                try:
                    relay.register()
                    worker.announce()
                except httpx.HTTPError as err:
                    print(f"[worker] could not create a new session: {err}")
                    time.sleep(5)
                continue
            failures += 1
            wait = min(60, 2 ** failures)
            print(f"[worker] relay error {e.response.status_code}; retrying in {wait}s")
            time.sleep(wait)
            continue
        except httpx.HTTPError as e:
            failures += 1
            wait = min(60, 2 ** failures)
            print(f"[worker] relay unreachable ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
            continue

        if message is None:
            continue
        if message["type"] == "approval":
            worker.approvals.deliver(message)
        elif message["type"] == "workspace":
            worker.switch_workspace(str(message.get("name", "")))
        elif message["type"] == "model":
            worker.switch_model(str(message.get("name", "")))
        elif message["type"] == "task":
            worker.start_task(str(message.get("text", "")))


if __name__ == "__main__":
    main()