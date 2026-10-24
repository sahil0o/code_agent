"""Relay server: passes messages between the phone (client) and the Mac (worker).

It never runs any code. It only stores messages and hands them to the other side.

Run locally:   RELAY_WORKER_KEY=... uvicorn relay:app --port 8000
On a server:   uvicorn relay:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips="*"
"""
import asyncio
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ---------- Settings ----------
WORKER_KEY = os.environ.get("RELAY_WORKER_KEY", "")
if len(WORKER_KEY) < 32:
    raise SystemExit("Set RELAY_WORKER_KEY to a long random secret (32+ characters). "
                     "Generate one with:  python -c \"import secrets; print(secrets.token_urlsafe(32))\"")

LONG_POLL_SECONDS = 25
SESSION_LIFETIME = int(os.environ.get("SESSION_LIFETIME", 12 * 3600))   # seconds
PAIRING_LIFETIME = int(os.environ.get("PAIRING_LIFETIME", 10 * 60))     # a pairing link works for 10 minutes...
MAX_SESSIONS = 50                    # ...and only once
MAX_EVENTS_PER_SESSION = 5000
MAX_INBOX = 100                      # messages waiting for the worker
MAX_BODY_BYTES = 256_000             # any single request
MAX_FAILURES_PER_IP = 20             # wrong tokens / codes allowed per IP...
FAILURE_WINDOW = 600                 # ...per 10 minutes
MAX_NEW_SESSIONS_PER_IP = 20         # per hour

PAGE = Path(__file__).parent / "static" / "index.html"


class Session:
    def __init__(self):
        self.worker_token = secrets.token_urlsafe(32)   # proves "I am the Mac"
        self.client_token = secrets.token_urlsafe(32)   # proves "I am the phone"; never put in a link
        self.created = time.time()
        self.events = []
        self.new_event = asyncio.Event()
        self.inbox = asyncio.Queue(maxsize=MAX_INBOX)
        self.new_pairing_code()

    def new_pairing_code(self):
        self.pairing_code = secrets.token_urlsafe(16)
        self.pairing_expires = time.time() + PAIRING_LIFETIME

    def expired(self):
        return time.time() - self.created > SESSION_LIFETIME


sessions = {}


# ---------- Rate limiting (in memory, per IP address) ----------
failures = defaultdict(deque)        # ip -> times of recent failed attempts
new_sessions = defaultdict(deque)    # ip -> times of recent session creations


def recent(times, window):
    """Drop entries older than the window; return how many remain."""
    now = time.time()
    while times and now - times[0] > window:
        times.popleft()
    return len(times)


def client_ip(request: Request):
    return request.client.host if request.client else "unknown"


def check_not_blocked(request: Request):
    if recent(failures[client_ip(request)], FAILURE_WINDOW) >= MAX_FAILURES_PER_IP:
        raise HTTPException(429, "Too many failed attempts. Try again later.")


def record_failure(request: Request, status, detail):
    failures[client_ip(request)].append(time.time())
    raise HTTPException(status, detail)


def get_session(request, session_id, token, role):
    """Look up a session and check the caller's token. role is 'worker' or 'client'."""
    check_not_blocked(request)
    session = sessions.get(session_id)
    if session is None or session.expired():
        record_failure(request, 404, "Unknown or expired session")
    expected = session.worker_token if role == "worker" else session.client_token
    if not token or not secrets.compare_digest(token, expected):
        record_failure(request, 403, "Wrong token")
    return session


# ---------- Cleanup of old sessions ----------
async def cleanup_loop():
    while True:
        await asyncio.sleep(60)
        for session_id in [sid for sid, s in sessions.items() if s.expired()]:
            del sessions[session_id]


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(cleanup_loop())   # runs in the background for the server's whole life
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)   # no public /docs page on the internet


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method == "POST":
        length = request.headers.get("content-length")
        if length is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        if int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request too large"}, status_code=413)
    return await call_next(request)


# ---------- Worker (the Mac) ----------
@app.post("/sessions")
def create_session(request: Request, x_worker_key: str = Header(None)):
    """Only someone holding RELAY_WORKER_KEY (your Mac) can create sessions."""
    check_not_blocked(request)
    if not x_worker_key or not secrets.compare_digest(x_worker_key, WORKER_KEY):
        record_failure(request, 401, "Wrong worker key")
    if recent(new_sessions[client_ip(request)], 3600) >= MAX_NEW_SESSIONS_PER_IP:
        raise HTTPException(429, "Too many new sessions. Try again later.")
    for session_id in [sid for sid, s in sessions.items() if s.expired()]:
        del sessions[session_id]
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(503, "Relay is full. Try again later.")

    new_sessions[client_ip(request)].append(time.time())
    session_id = secrets.token_urlsafe(8)
    session = Session()
    sessions[session_id] = session
    return {"session_id": session_id, "worker_token": session.worker_token,
            "pairing_code": session.pairing_code, "pairing_seconds": PAIRING_LIFETIME}


@app.post("/sessions/{session_id}/pairing")
def new_pairing(session_id: str, request: Request, authorization: str = Header(None)):
    """The worker asks for a fresh pairing code, e.g. to connect another device."""
    session = get_session(request, session_id, authorization, "worker")
    session.new_pairing_code()
    return {"pairing_code": session.pairing_code, "pairing_seconds": PAIRING_LIFETIME}


@app.get("/sessions/{session_id}/inbox")
async def worker_inbox(session_id: str, request: Request, authorization: str = Header(None)):
    session = get_session(request, session_id, authorization, "worker")
    try:
        message = await asyncio.wait_for(session.inbox.get(), timeout=LONG_POLL_SECONDS)
        return {"message": message}
    except asyncio.TimeoutError:
        return {"message": None}


@app.post("/sessions/{session_id}/events")
def worker_event(session_id: str, event: dict, request: Request, authorization: str = Header(None)):
    session = get_session(request, session_id, authorization, "worker")
    if len(session.events) >= MAX_EVENTS_PER_SESSION:
        raise HTTPException(507, "This session is full. Restart the worker for a new one.")
    event["index"] = len(session.events)
    session.events.append(event)
    session.new_event.set()
    session.new_event = asyncio.Event()
    return {"ok": True, "index": event["index"]}


# ---------- Client (the phone) ----------
@app.post("/sessions/{session_id}/pair")
def pair(session_id: str, body: dict, request: Request):
    """Trade a one-time pairing code for the long-lived client token."""
    check_not_blocked(request)
    session = sessions.get(session_id)
    code = str(body.get("code", ""))
    if (session is None or session.expired() or not session.pairing_code
            or time.time() > session.pairing_expires
            or not secrets.compare_digest(code, session.pairing_code)):
        record_failure(request, 403, "This pairing link is invalid, expired or already used.")
    session.pairing_code = None              # one use only
    return {"client_token": session.client_token}


@app.post("/sessions/{session_id}/messages")
async def client_message(session_id: str, message: dict, request: Request, authorization: str = Header(None)):
    session = get_session(request, session_id, authorization, "client")
    if message.get("type") not in ("task", "approval", "workspace", "model"):
        raise HTTPException(400, "type must be 'task', 'approval', 'workspace' or 'model'")
    if len(str(message.get("text", ""))) > 20_000:
        raise HTTPException(413, "Task text too long")
    try:
        session.inbox.put_nowait(message)
    except asyncio.QueueFull:
        raise HTTPException(429, "Too many messages waiting. Is the worker running?")
    return {"ok": True}


@app.get("/sessions/{session_id}/events")
async def client_events(session_id: str, request: Request, after: int = -1, authorization: str = Header(None)):
    session = get_session(request, session_id, authorization, "client")
    if len(session.events) <= after + 1:
        waiter = session.new_event
        try:
            await asyncio.wait_for(waiter.wait(), timeout=LONG_POLL_SECONDS)
        except asyncio.TimeoutError:
            pass
    return {"events": session.events[after + 1:]}


# ---------- The web page ----------
@app.get("/", response_class=HTMLResponse)
def web_page():
    return PAGE.read_text()


@app.get("/health")
def health():
    """For the hosting platform's health checks."""
    return {"ok": True}