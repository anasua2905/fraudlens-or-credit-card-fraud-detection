"""On GitHub Actions, repeat an uncaught error (and the output just before it) as an annotation.

The run page shows annotations without opening the log, so a failing step explains itself.

    import ci_annotate; ci_annotate.install("pipeline")
"""
from __future__ import annotations

import collections
import os
import sys
import traceback

_TAIL: collections.deque[str] = collections.deque(maxlen=30)


class _Tee:
    def __init__(self, stream):
        self._stream, self._buf = stream, ""

    def write(self, text):
        self._buf += text
        *lines, self._buf = self._buf.split("\n")
        _TAIL.extend(ln for ln in lines if ln.strip())
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def install(title: str) -> None:
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    sys.stdout, sys.stderr = _Tee(sys.stdout), _Tee(sys.stderr)
    previous = sys.excepthook

    def hook(etype, value, tb):
        previous(etype, value, tb)
        trace = "".join(traceback.format_exception(etype, value, tb)).strip().splitlines()
        lines = list(_TAIL)[-12:] + ["--- traceback (last lines)"] + trace[-14:]
        sys.__stdout__.write(f"::error title={title} failed::{_escape(chr(10).join(lines))[:4000]}\n")
        sys.__stdout__.flush()

    sys.excepthook = hook


def run(main, title: str) -> None:
    """Run main(); on a non-zero exit (returned, SystemExit or error) annotate the output tail."""
    install(title)
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code
        if code not in (None, 0) and not isinstance(code, int):
            print(code, file=sys.stderr)
            code = 1
    if code not in (None, 0) and os.environ.get("GITHUB_ACTIONS"):
        sys.__stdout__.write(f"::error title={title} failed (exit {code})::"
                             f"{_escape(chr(10).join(list(_TAIL)[-25:]))[:4000]}\n")
        sys.__stdout__.flush()
    sys.exit(code or 0)
