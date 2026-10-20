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

import agent      # also loads .env, so RELAY_WORKER_KEY can live there
import tools
from logger import RunLogger

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
        self.relay.send("approval_needed", id=approval_id, diff=diff_text)
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


def load_workspaces():
    """Folders the phone may switch between, set ON THE MAC in .env, e.g.
    AGENT_WORKSPACES=practice=~/code/playground,textkit=~/code/textkit
    The phone only ever sends a NAME from this list, never a path."""
    spaces = {tools.WORKSPACE.name: tools.WORKSPACE}          # the starting workspace is always allowed
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


def announce(relay, workspaces):
    relay.send("worker_online", workspace=current_workspace_name(workspaces), workspaces=sorted(workspaces))


def current_workspace_name(workspaces):
    return next((name for name, folder in workspaces.items() if folder == tools.WORKSPACE), tools.WORKSPACE.name)


def switch_workspace(relay, workspaces, name, busy):
    if busy:
        relay.send("notice", text="Can't switch workspace while a task is running.")
    elif name not in workspaces:
        relay.send("notice", text=f"Unknown workspace '{name}'. Allowed: {', '.join(sorted(workspaces))}")
    else:
        tools.WORKSPACE = workspaces[name]
        print(f"[worker] workspace switched to {name}: {tools.WORKSPACE}")
        relay.send("workspace_changed", workspace=name)
        return
    relay.send("workspace_changed", workspace=current_workspace_name(workspaces))   # tell the page what is still active


def run_task(relay, text):
    """Runs in a background thread, so the main loop keeps receiving approvals meanwhile."""
    relay.send("task_started", text=text)
    try:
        log = RelayLogger(relay, text, agent.MODEL, tools.WORKSPACE)
        answer = agent.run_agent(text, log=log)
        relay.send("answer", text=answer, summary=log.record.get("summary", {}))
    except Exception as e:
        relay.send("error", text=f"{type(e).__name__}: {e}")


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
    relay = RelayConnection(RELAY_URL)
    approvals = RemoteApprovals(relay)
    tools.ask_approval = approvals.ask

    workspaces = load_workspaces()
    print(f"[worker] relay: {RELAY_URL}")
    print(f"[worker] workspaces the phone may choose: {', '.join(sorted(workspaces))}")
    print(f"[worker] current workspace: {tools.WORKSPACE}\n")
    relay.register()
    announce(relay, workspaces)
    if sys.stdin.isatty():
        threading.Thread(target=watch_keyboard, args=(relay,), daemon=True).start()

    current_task = None
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
                    announce(relay, workspaces)
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
        busy = bool(current_task and current_task.is_alive())
        if message["type"] == "approval":
            approvals.deliver(message)
        elif message["type"] == "workspace":
            switch_workspace(relay, workspaces, str(message.get("name", "")), busy)
        elif message["type"] == "task":
            if busy:
                relay.send("notice", text="Busy with another task. Try again when it finishes.")
                continue
            print(f"[worker] new task: {message.get('text')}")
            current_task = threading.Thread(target=run_task, args=(relay, message.get("text", "")), daemon=True)
            current_task.start()


if __name__ == "__main__":
    main()