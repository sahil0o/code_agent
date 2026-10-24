import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from collections import Counter
RUNS_DIR = Path(__file__).parent / "runs"


class RunLogger:
    """Records everything one agent run does, and saves it as a JSON file in runs/."""

    def __init__(self, question, model, workspace):
        self.started = time.time()
        # timestamp + short random suffix, so two runs in the same second never share a file
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.record = {
            "run_id": self.run_id,
            "question": question,
            "model": model,
            "workspace": str(workspace),
            "events": [],
        }

    def _add(self, event_type, **data):
        data["type"] = event_type
        data["t"] = round(time.time() - self.started, 2)   # seconds since the run started
        self.record["events"].append(data)

    def model_call(self, step, usage, seconds, finish_reason):
        self._add("model_call", step=step, seconds=round(seconds, 2), finish_reason=finish_reason,
                  tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                  tokens_out=getattr(usage, "completion_tokens", 0) or 0)

    def tool_call(self, step, name, arguments, result):
        self._add("tool_call", step=step, name=name, arguments=arguments,
                  is_error=result.startswith("Error"), result_chars=len(result),
                  result_preview=result[:300])

    def api_error(self, step, code, failed_generation):
        self._add("api_error", step=step, code=code, failed_generation=str(failed_generation)[:1000])

    def finish(self, outcome, answer):
        """outcome is one of: 'answered', 'step_limit', 'crashed'."""
        events = self.record["events"]
        calls = [e for e in events if e["type"] == "model_call"]
        tools = [e for e in events if e["type"] == "tool_call"]
        self.record["answer"] = answer
        self.record["summary"] = {
            "outcome": outcome,
            "steps": max((e["step"] for e in events if "step" in e), default=0),
            "model_calls": len(calls),
            "tool_calls": len(tools),
            "tools_used": dict(Counter(e["name"] for e in tools)),
            "tool_errors": sum(e["is_error"] for e in tools),
            "api_errors": sum(e["type"] == "api_error" for e in events),
            "tokens_in": sum(e["tokens_in"] for e in calls),
            "tokens_out": sum(e["tokens_out"] for e in calls),
            "seconds": round(time.time() - self.started, 1),
        }
        RUNS_DIR.mkdir(exist_ok=True)
        path = RUNS_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(self.record, indent=2, ensure_ascii=False))
        return path