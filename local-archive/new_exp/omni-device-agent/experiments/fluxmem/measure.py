"""Instrumentation only: sampled RSS/MPS peaks and synchronized wall time."""
from contextlib import contextmanager
import threading
import time
import psutil
import ctypes
import sys
import json

class RUsageInfoV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in [
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
        "wired_size", "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime"]]


class Meter:
    def __init__(self, torch, device='mps', interval=.01, trace_path=None):
        self.trace = open(trace_path, "x") if trace_path is not None else None
        self.context = {}
        self.torch = torch
        self.device = device
        self.interval = interval
        self.process = psutil.Process()
        self.libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None
        if self.libproc:
            self.libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self.libproc.proc_pid_rusage.restype = ctypes.c_int
        self.active = []
        self.events = []
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def sync(self):
        if self.device == 'mps':
            self.torch.mps.synchronize()

    def snapshot(self):
        result = {'rss_bytes': self.process.memory_info().rss}
        if self.libproc:
            usage=RUsageInfoV0()
            if self.libproc.proc_pid_rusage(self.process.pid,0,ctypes.byref(usage)) != 0:
                raise RuntimeError('proc_pid_rusage failed')
            result['physical_footprint_bytes']=usage.phys_footprint
        if self.device == 'mps':
            result.update(mps_allocated_bytes=self.torch.mps.current_allocated_memory(),
                          mps_driver_bytes=self.torch.mps.driver_allocated_memory())
        return result

    def _sample(self):
        values = self.snapshot()
        with self.lock:
            for event in self.active:
                for k, v in values.items():
                    event['sampled_peak'][k] = max(event['sampled_peak'].get(k, 0), v)
                event['samples'] += 1

    def _poll(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def begin(self, name, **details):
        self.sync()
        before = self.snapshot()
        event = {'stage': name, 'before': before, 'sampled_peak': dict(before),
                 'samples': 1, **self.context, **details, '_start': time.perf_counter()}
        with self.lock:
            self.active.append(event)
            if self.trace:
                self.trace.write(json.dumps({"event":"begin","stage":name,**self.context})+"\n");self.trace.flush()
        return event

    def end(self, event, **details):
        self.sync()
        elapsed = time.perf_counter() - event['_start']
        self._sample()
        with self.lock:
            self.active.remove(event)
            event.pop('_start')
            event.update(seconds=elapsed, after=self.snapshot(), **details)
            self.events.append(event)
            if self.trace:
                self.trace.write(json.dumps({"event":"end",**event})+"\n");self.trace.flush()
        return event

    @contextmanager
    def phase(self, name, **details):
        event = self.begin(name, **details)
        try:
            yield event
        finally:
            self.end(event)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.trace:self.trace.close()
