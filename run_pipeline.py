"""Non-interruptible structured annotation pipeline runner.

Runs annotate_page_structured() over a directory of processed images in its
own Windows process group so that Ctrl+C in the monitoring terminal does NOT
kill it.  Progress is written to a log file; use watch_pipeline.py to tail it.

Usage
-----
Start a run (returns immediately, pipeline runs in the background)::

    uv run python run_pipeline.py start --input data/processed

Watch live progress::

    uv run python run_pipeline.py watch

Stop the running pipeline (SIGTERM)::

    uv run python run_pipeline.py stop

Resume an interrupted run::

    uv run python run_pipeline.py start --input data/processed  # checkpoint picks up automatically
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_PID_FILE = Path("data/interim/.pipeline.pid")
_LOG_FILE = Path("data/interim/pipeline.log")
_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Sub-command: start
# ---------------------------------------------------------------------------

def _cmd_start(args: argparse.Namespace) -> None:
    if _PID_FILE.exists():
        pid = int(_PID_FILE.read_text())
        # Check if process still running
        try:
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x400, False, pid)  # PROCESS_QUERY_INFORMATION
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                print(f"Pipeline already running (PID {pid}). Use 'stop' first or 'watch' to monitor.")
                return
        except Exception:
            pass  # Process not found; stale pid file
        _PID_FILE.unlink(missing_ok=True)

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _PYTHON, "-m", "newspapers.segmentation.annotate",
        "--input", str(args.input),
        "--labels", str(args.labels),
        "--images", str(args.images),
        "--vis", str(args.vis),
        "--generator-model", args.generator_model,
        "--critic-model", args.critic_model,
        "--critique-rounds", str(args.critique_rounds),
        "--n-columns", str(args.n_columns),
        "--overlap-frac", str(args.overlap_frac),
        "--structured",
        "--verbose",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.show_vis:
        cmd.append("--show-vis")

    log_fh = open(_LOG_FILE, "a", encoding="utf-8", buffering=1)

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        # CREATE_NEW_PROCESS_GROUP detaches from the parent's console; Ctrl+C
        # in this terminal won't propagate to the child.
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        close_fds=True,
    )

    _PID_FILE.write_text(str(proc.pid))
    print(f"Pipeline started (PID {proc.pid}). Logging to {_LOG_FILE}")
    print(f"Run 'uv run python run_pipeline.py watch' to follow progress.")


# ---------------------------------------------------------------------------
# Sub-command: watch
# ---------------------------------------------------------------------------

def _cmd_watch(args: argparse.Namespace) -> None:  # noqa: ARG001
    if not _LOG_FILE.exists():
        print(f"No log file found at {_LOG_FILE}. Has a pipeline been started?")
        return

    print(f"Watching {_LOG_FILE}  (Ctrl+C to stop watching — pipeline keeps running)\n")
    try:
        with open(_LOG_FILE, encoding="utf-8") as fh:
            # Print existing content first
            print(fh.read(), end="", flush=True)
            # Then tail new lines
            while True:
                line = fh.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopped watching. Pipeline is still running in the background.")


# ---------------------------------------------------------------------------
# Sub-command: stop
# ---------------------------------------------------------------------------

def _cmd_stop(args: argparse.Namespace) -> None:  # noqa: ARG001
    if not _PID_FILE.exists():
        print("No PID file found. Is a pipeline running?")
        return

    pid = int(_PID_FILE.read_text())
    import ctypes
    PROCESS_TERMINATE = 0x0001
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
    if not handle:
        print(f"Could not find process {pid}. May have already finished.")
        _PID_FILE.unlink(missing_ok=True)
        return

    ctypes.windll.kernel32.TerminateProcess(handle, 1)
    ctypes.windll.kernel32.CloseHandle(handle)
    _PID_FILE.unlink(missing_ok=True)
    print(f"Pipeline (PID {pid}) terminated.")


# ---------------------------------------------------------------------------
# Sub-command: status
# ---------------------------------------------------------------------------

def _cmd_status(args: argparse.Namespace) -> None:  # noqa: ARG001
    if not _PID_FILE.exists():
        print("No pipeline running (no PID file).")
        return

    pid = int(_PID_FILE.read_text())
    print(f"PID file: {pid}")

    # Count completed/total strips from any checkpoint files
    vis_dir = Path("data/annotations/visualizations")
    checkpoints = list(vis_dir.glob("*_checkpoint.json")) if vis_dir.exists() else []
    if checkpoints:
        for cp in checkpoints:
            try:
                data = json.loads(cp.read_text(encoding="utf-8"))
                done = len(data.get("completed_strips", []))
                stem = cp.stem.replace("_checkpoint", "")
                print(f"  {stem}: {done} strips done")
            except Exception:
                pass
    else:
        print("  No checkpoint files found yet (pipeline may not have started processing).")

    if _LOG_FILE.exists():
        # Show last 5 lines of log
        lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
        print(f"\nLast log entries ({_LOG_FILE}):")
        for line in lines[-5:]:
            print(f"  {line}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Non-interruptible structured annotation pipeline runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- start ---
    s = sub.add_parser("start", help="Launch the pipeline in the background.")
    s.add_argument("--input", type=Path, default=Path("data/processed"),
                   help="Input directory of .jpg images.")
    s.add_argument("--labels", type=Path, default=Path("data/annotations/labels/train"))
    s.add_argument("--images", type=Path, default=Path("data/annotations/images/train"))
    s.add_argument("--vis", type=Path, default=Path("data/annotations/visualizations"))
    s.add_argument("--generator-model", default="gemini-2.5-flash")
    s.add_argument("--critic-model", default="gemini-2.5-pro")
    s.add_argument("--critique-rounds", type=int, default=1)
    s.add_argument("--n-columns", type=int, default=8)
    s.add_argument("--overlap-frac", type=float, default=0.05)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--show-vis", action="store_true",
                   help="Open each visualisation PNG after it is written.")
    s.set_defaults(func=_cmd_start)

    # --- watch ---
    w = sub.add_parser("watch", help="Tail the pipeline log (pipeline keeps running on Ctrl+C).")
    w.set_defaults(func=_cmd_watch)

    # --- stop ---
    st = sub.add_parser("stop", help="Terminate the running pipeline.")
    st.set_defaults(func=_cmd_stop)

    # --- status ---
    ss = sub.add_parser("status", help="Show pipeline status and checkpoint progress.")
    ss.set_defaults(func=_cmd_status)

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    args.func(args)
