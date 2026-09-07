"""Task-local wall accounting. Concurrent work never counts twice in the total."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import inspect
from threading import Lock
import time

_CURRENT = ContextVar("omniflow_timing_span", default=None)


class TimingLedger:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self._spans = []
        self._lock = Lock()

    @contextmanager
    def span(self, name):
        current = _CURRENT.get()
        parent = current[1] if current and current[0] is self else None
        with self._lock:
            index = len(self._spans)
            span = {"name": name, "parent": parent, "start": self.clock(), "end": None, "failed": False}
            self._spans.append(span)
        token = _CURRENT.set((self, index))
        try:
            yield
        except BaseException:
            span["failed"] = True
            raise
        finally:
            span["end"] = self.clock()
            _CURRENT.reset(token)

    def report(self):
        with self._lock:
            spans = [dict(span) for span in self._spans]
        if any(s["end"] is None for s in spans):
            raise RuntimeError("timing_report_requires_drained_spans")
        components = defaultdict(lambda: {"calls": 0, "failed_calls": 0, "inclusive_ms": 0.0, "exclusive_ms": 0.0})
        events = defaultdict(list)
        for index, span in enumerate(spans):
            item = components[span["name"]]
            item["calls"] += 1
            item["failed_calls"] += int(span["failed"])
            item["inclusive_ms"] += (span["end"] - span["start"]) / 1e6
            events[span["start"]].append((index, True))
            events[span["end"]].append((index, False))
        active = set()
        previous = None
        total = 0.0
        for tick in sorted(events):
            if previous is not None and active:
                parents = {spans[i]["parent"] for i in active}
                leaves = active - parents
                name = spans[next(iter(leaves))]["name"] if len(leaves) == 1 else "concurrent"
                elapsed = (tick - previous) / 1e6
                components[name]["exclusive_ms"] += elapsed
                total += elapsed
            for index, opening in events[tick]:
                if opening:
                    active.add(index)
                else:
                    active.discard(index)
            previous = tick
        return {"schema_version": "omniflow.wall-accounting.v1", "covered_wall_ms": total,
                "accounted_wall_ms": sum(v["exclusive_ms"] for v in components.values()),
                "components": dict(components),
                "semantics": "exclusive partitions wall time; inclusive spans overlap and must not be summed"}


def measure(name):
    current = _CURRENT.get()
    return current[0].span(name) if current else nullcontext()


def timed(name):
    def decorate(callback):
        if inspect.iscoroutinefunction(callback):
            @wraps(callback)
            async def asynchronous(*args, **kwargs):
                with measure(name):
                    return await callback(*args, **kwargs)
            return asynchronous
        @wraps(callback)
        def synchronous(*args, **kwargs):
            with measure(name):
                return callback(*args, **kwargs)
        return synchronous
    return decorate
