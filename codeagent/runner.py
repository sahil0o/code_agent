"""How commands actually get executed. The agent's run_command tool decides WHAT may run;
a Runner decides HOW it runs. Today: LocalRunner (on this computer). Next: a DockerRunner (in a sandbox)."""
import os
import subprocess
from dataclasses import dataclass

# Environment variables whose names contain any of these words are removed before running a command,
# so code the model wrote cannot read your API keys (GROQ_API_KEY, RELAY_WORKER_KEY, ...).
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass
class CommandResult:
    exit_code: int | None   # None when the command was stopped by the timeout
    stdout: str
    stderr: str
    timed_out: bool = False


def safe_environment():
    env = {name: value for name, value in os.environ.items()
           if not any(word in name.upper() for word in SECRET_WORDS)}
    env["PYTHONUTF8"] = "1"          # UTF-8 everywhere, also on Windows
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"        # matplotlib saves files instead of opening windows (which would hang)
    return env


def as_text(output):
    """subprocess gives bytes instead of text when a timeout interrupts it."""
    if output is None:
        return ""
    return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output


class LocalRunner:
    """Runs the command directly on this computer, inside the workspace folder."""
    name = "local"

    def run(self, args, cwd, timeout):
        try:
            proc = subprocess.run(
                args,                          # a list, never a string: no shell is involved
                cwd=cwd,
                env=safe_environment(),
                stdin=subprocess.DEVNULL,      # scripts that wait for input() fail fast instead of hanging
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=timeout,
            )
            return CommandResult(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as e:
            return CommandResult(None, as_text(e.stdout), as_text(e.stderr), timed_out=True)