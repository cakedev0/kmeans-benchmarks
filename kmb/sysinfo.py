"""Library versions, hardware description, and GPU sampling through NVML (ctypes, no deps)."""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
import socket
import threading
import time


def versions(distributions) -> dict:
    """Installed versions from package metadata (not stale ``__version__`` attributes)."""
    out = {"python": platform.python_version()}
    for dist in distributions:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def host_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "kernel": platform.release(),
    }
    try:
        with open("/proc/meminfo") as f:
            info["mem_total_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        pass
    return info


# --- NVML -------------------------------------------------------------------------------

class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class NVML:
    """Minimal NVML binding for device 0 (or ``CUDA_VISIBLE_DEVICES``'s first device)."""

    NVML_CLOCK_SM = 1
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, index=0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit failed")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
            raise RuntimeError("no NVML device")

    def _str(self, fn, *args, size=96):
        buf = ctypes.create_string_buffer(size)
        fn(*args, buf, size)
        return buf.value.decode()

    def static_info(self) -> dict:
        lib, h = self.lib, self.handle
        info = {
            "gpu_name": self._str(lib.nvmlDeviceGetName, h),
            "driver": self._str(lib.nvmlSystemGetDriverVersion),
        }
        cuda = ctypes.c_int()
        if lib.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(cuda)) == 0:
            info["cuda_driver"] = f"{cuda.value // 1000}.{cuda.value % 1000 // 10}"
        value = ctypes.c_uint()
        if lib.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(value)) == 0:
            info["power_limit_w"] = value.value / 1000
        if lib.nvmlDeviceGetMaxClockInfo(h, self.NVML_CLOCK_SM, ctypes.byref(value)) == 0:
            info["sm_clock_max_mhz"] = value.value
        info["memory_total_gb"] = round(self.memory_used_total()[1] / 2**30, 1)
        return info

    def memory_used_total(self):
        mem = _Memory()
        self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
        return mem.used, mem.total

    def sample(self) -> dict:
        lib, h = self.lib, self.handle
        value = ctypes.c_uint()
        out = {"mem_used": self.memory_used_total()[0]}
        if lib.nvmlDeviceGetClockInfo(h, self.NVML_CLOCK_SM, ctypes.byref(value)) == 0:
            out["sm_clock_mhz"] = value.value
        if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(value)) == 0:
            out["power_w"] = value.value / 1000
        if lib.nvmlDeviceGetTemperature(h, self.NVML_TEMPERATURE_GPU, ctypes.byref(value)) == 0:
            out["temp_c"] = value.value
        return out


def gpu_info() -> dict | None:
    try:
        return NVML().static_info()
    except (OSError, RuntimeError):
        return None


class GPUSampler:
    """Samples device memory, SM clock, power and temperature in a background thread.

    Peak memory is device-wide used memory minus the value at ``start`` (the machine is
    locked, so nothing else allocates). Samples every ``interval`` seconds, so allocations
    shorter than that can be missed. The thread sleeps between ctypes calls, which release
    the GIL, so it barely perturbs the timed code.
    """

    def __init__(self, interval=0.02):
        self.nvml = NVML()
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.baseline = self.nvml.memory_used_total()[0]
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self.nvml.sample())
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        self.samples.append(self.nvml.sample())
        s = self.samples
        out = {"peak_mem_bytes": max(x["mem_used"] for x in s) - self.baseline,
               "peak_mem_method": "nvml device used memory, sampled every "
                                  f"{int(self.interval * 1000)} ms, minus baseline"}
        for key in ("sm_clock_mhz", "power_w", "temp_c"):
            values = [x[key] for x in s if key in x]
            if values:
                out[f"gpu_{key}_mean"] = round(sum(values) / len(values), 1)
        out["gpu_temp_c_max"] = max((x.get("temp_c", 0) for x in s), default=None)
        return out


# --- CPU peak memory -----------------------------------------------------------------------

def reset_peak_rss() -> bool:
    """Reset the kernel's peak RSS counter (VmHWM) of this process (Linux >= 4.0)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _status_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return int(line.split()[1]) * 1024
    return None


def rss_bytes():
    return _status_kb("VmRSS:")


def peak_rss_bytes():
    return _status_kb("VmHWM:")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
