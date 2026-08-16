#!/usr/bin/env python3
"""
Tail a YOLOv6 training log and emit only meaningful lines:
  - completed epoch summaries (no tqdm progress bar garbage)
  - mAP / AP50 / Evaluating lines
  - errors (Error, Traceback, Killed, OOM)

Usage: python3 tools/monitor_train.py <logfile> [total_epochs]
"""
import sys
import re
import time

LOG = sys.argv[1] if len(sys.argv) > 1 else "runs/train/roi200to640_train.log"
TOTAL = sys.argv[2] if len(sys.argv) > 2 else "79"

# Strict epoch line: leading spaces, "N/TOTAL", exactly 5 floats, end of string
# Example: "     13/79   0.001892   0.4818   0.4872   0.6989   0.8765"
RE_EPOCH = re.compile(
    r"^\s+\d+/" + TOTAL + r"(\s+\d+\.\d+){5}\s*$"
)
RE_EVAL  = re.compile(r"mAP|AP50|Evaluating", re.IGNORECASE)
RE_SKIP  = re.compile(r"Loading",             re.IGNORECASE)
RE_ERROR = re.compile(r"Error|Traceback|Killed|OOM", re.IGNORECASE)


def process(raw: bytes) -> None:
    """Split a raw chunk on both \r and \n, filter, and print matching lines."""
    text = raw.decode("utf-8", errors="replace")
    for line in re.split(r"[\r\n]", text):
        line = line.rstrip()
        if not line:
            continue
        if RE_EPOCH.match(line):
            print(line, flush=True)
        elif RE_EVAL.search(line) and not RE_SKIP.search(line):
            print(line, flush=True)
        elif RE_ERROR.search(line):
            print(line, flush=True)


buf = b""
with open(LOG, "rb") as f:
    f.seek(0, 2)          # start at end of file
    while True:
        chunk = f.read(131072)
        if chunk:
            buf += chunk
            # split on newlines only; \r is handled inside process()
            parts = buf.split(b"\n")
            buf = parts[-1]           # keep incomplete last line
            for part in parts[:-1]:
                process(part + b"\n")
        else:
            time.sleep(1)
