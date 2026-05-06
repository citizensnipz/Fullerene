from __future__ import annotations

import queue
import threading
from typing import Protocol


class InputProvider(Protocol):
    def poll_line(self, timeout_seconds: float) -> str | None: ...

    def close(self) -> None: ...


class ThreadedStdinInputProvider:
    """Single-purpose input thread that only moves full lines into a queue."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                line = input()
            except EOFError:
                self._queue.put("__EOF__")
                return
            except Exception:
                self._queue.put("__INPUT_ERROR__")
                return
            self._queue.put(line)

    def poll_line(self, timeout_seconds: float) -> str | None:
        timeout = max(0.0, float(timeout_seconds))
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if line == "__EOF__":
            return "/quit"
        if line == "__INPUT_ERROR__":
            return "/quit"
        return line

    def close(self) -> None:
        self._stop.set()
