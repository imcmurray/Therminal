#!/usr/bin/env python3
"""Therminal — beautiful GPU + CPU thermal + load TUI."""

import argparse
import csv
from collections import Counter, deque

# Temperature smoothing to reduce jumpiness in displayed values and per-core history
_TEMP_SMOOTHING_WINDOW = 300  # ~5 minutes of averaging at ~1s refresh (tunable)
_package_temp_history: deque[float] = deque(maxlen=_TEMP_SMOOTHING_WINDOW)
_core_temp_histories: dict[str, deque[float]] = {}  # physical core label -> recent readings

# History window for tracking which physical cores have been hottest over time
# (used to color the C0-C7 labels in the CORES section)
_HOTTEST_HISTORY_WINDOW = 1000  # sizeable sampling range for stable long-term view
import glob
import os
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

import psutil


console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GpuInfo:
    index: int
    name: str
    util: float          # 0-100
    mem_used: float      # MB
    mem_total: float     # MB
    temp: float          # °C
    power: float         # W (current)
    power_limit: float   # W
    fan: float           # % or -1 if unavailable
    sm_count: int = 0    # Streaming Multiprocessor count
    cuda_cores: int = 0  # Total CUDA cores (estimated or looked up)
    sm_clock_mhz: int = 0          # Current SM clock speed
    throttle_reasons: str = ""     # Human readable throttle status


@dataclass
class GpuProcess:
    pid: int
    name: str
    gpu_index: int
    mem_mib: float
    mem_pct: float          # percentage of that GPU's total memory
    sm_util: float = 0.0    # SM utilization % from pmon (0-100)
    mem_util: float = 0.0   # Memory bandwidth util % from pmon (0-100)


@dataclass
class CpuProcess:
    name: str
    cpu: float              # CPU % used by this process (0-100+)
    mem_mib: float          # RSS memory in MiB


@dataclass
class CpuSnapshot:
    overall: float
    per_core: list[float]
    freq_mhz: float
    temp_package: float | None
    temp_cores: list[tuple[str, float]]   # (label, temperature)
    load1: float
    load5: float
    load15: float
    uptime: timedelta


# ──────────────────────────────────────────────────────────────────────────────
# Color helpers
# ──────────────────────────────────────────────────────────────────────────────

def util_color(pct: float) -> str:
    if pct < 50:
        return "green"
    if pct < 80:
        return "yellow"
    return "red"


def temp_color(temp: float, is_gpu: bool = False) -> str:
    if is_gpu:
        if temp < 65:
            return "green"
        if temp < 80:
            return "yellow"
        return "red"
    else:
        if temp < 70:
            return "green"
        if temp < 85:
            return "yellow"
        return "red"


def load_color(load: float, cores: int) -> str:
    ratio = load / cores
    if ratio < 0.6:
        return "green"
    if ratio < 1.0:
        return "yellow"
    return "red"


def power_color(current: float, limit: float) -> str:
    if limit <= 0:
        return "cyan"
    ratio = current / limit
    if ratio < 0.7:
        return "green"
    if ratio < 0.9:
        return "yellow"
    return "red"


# ──────────────────────────────────────────────────────────────────────────────
# Bar rendering (beautiful unicode)
# ──────────────────────────────────────────────────────────────────────────────

def make_bar(value: float, width: int = 18, color: str = "green", label: str = "") -> Text:
    """Pure Text progress bar — extremely reliable layout."""
    clamped = max(0.0, min(100.0, value))
    filled = int(clamped / 100 * width)

    bar = Text()
    if filled > 0:
        bar.append("█" * filled, style=color)
    if width - filled > 0:
        bar.append("░" * (width - filled), style="dim")

    pct = Text(f" {clamped:5.1f}%", style=f"bold {color}")

    if label:
        lbl = Text(f"{label} ", style="bold dim")
        return Text.assemble(lbl, bar, pct)
    return Text.assemble(bar, pct)


def small_bar(value: float, width: int = 8) -> Text:
    """Compact bar for per-core display."""
    clamped = max(0.0, min(100.0, value))
    filled = int(clamped / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    color = util_color(clamped)
    return Text(bar, style=color)


def make_memory_bar(value: float, width: int = 18) -> Text:
    """Simple memory bar for process list (no percentage text)."""
    clamped = max(0.0, min(100.0, value))
    filled = int(clamped / 100 * width)
    bar = Text()
    if filled > 0:
        bar.append("█" * filled, style=util_color(clamped))
    if width - filled > 0:
        bar.append("░" * (width - filled), style="dim")
    return bar


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────

def get_uptime() -> timedelta:
    with open("/proc/uptime") as f:
        seconds = float(f.readline().split()[0])
    return timedelta(seconds=int(seconds))


def get_loadavg() -> tuple[float, float, float]:
    with open("/proc/loadavg") as f:
        parts = f.readline().split()
    return float(parts[0]), float(parts[1]), float(parts[2])


def get_cpu_snapshot() -> CpuSnapshot:
    # CPU usage (short interval for responsiveness)
    overall = psutil.cpu_percent(interval=0.15)
    per_core = psutil.cpu_percent(percpu=True, interval=0.0)

    # Frequency
    freq = psutil.cpu_freq()
    freq_mhz = freq.current if freq else 0.0

    # Temperatures via psutil (with heavy smoothing for stable display)
    temps = psutil.sensors_temperatures()
    raw_temp_package = None
    raw_temp_cores: list[tuple[str, float]] = []

    if "coretemp" in temps:
        for entry in temps["coretemp"]:
            label_lower = (entry.label or "").lower()
            if "package" in label_lower or entry.label == "Package id 0":
                raw_temp_package = entry.current
            elif "core" in label_lower:
                core_label = entry.label or f"Core {len(raw_temp_cores)}"
                raw_temp_cores.append((core_label, entry.current))

    # Update smoothing buffers
    if raw_temp_package is not None:
        _package_temp_history.append(raw_temp_package)

    for label, temp in raw_temp_cores:
        if label not in _core_temp_histories:
            _core_temp_histories[label] = deque(maxlen=_TEMP_SMOOTHING_WINDOW)
        _core_temp_histories[label].append(temp)

    # Compute smoothed values
    temp_package = None
    if _package_temp_history:
        temp_package = sum(_package_temp_history) / len(_package_temp_history)

    temp_cores: list[tuple[str, float]] = []
    for label in [lbl for lbl, _ in raw_temp_cores]:  # preserve order
        hist = _core_temp_histories.get(label)
        if hist:
            smoothed = sum(hist) / len(hist)
            temp_cores.append((label, smoothed))

    load1, load5, load15 = get_loadavg()
    uptime = get_uptime()

    return CpuSnapshot(
        overall=overall,
        per_core=per_core,
        freq_mhz=freq_mhz,
        temp_package=temp_package,
        temp_cores=temp_cores,
        load1=load1,
        load5=load5,
        load15=load15,
        uptime=uptime,
    )


def parse_nvidia_smi() -> list[GpuInfo]:
    """Return GPU data or empty list if nvidia-smi unavailable or fails."""
    try:
        # Main reliable query (this was working before)
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw,power.limit,fan.speed,"
                "clocks.current.sm,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return []

        gpus: list[GpuInfo] = []
        reader = csv.reader(result.stdout.strip().splitlines())
        for row in reader:
            if len(row) < 11:
                continue
            idx = int(row[0])
            name = row[1].strip()
            util = float(row[2])
            mem_used = float(row[3])
            mem_total = float(row[4])
            temp = float(row[5])
            power = float(row[6]) if row[6].strip() else 0.0
            power_limit = float(row[7]) if row[7].strip() else 0.0
            fan = float(row[8]) if row[8].strip() and row[8].strip() != "-1" else -1.0
            sm_clock = int(row[9]) if len(row) > 9 and row[9].strip().isdigit() else 0
            throttle_raw = row[10].strip() if len(row) > 10 else ""

            # Convert throttle bitmask to readable string
            throttle_str = ""
            if throttle_raw and throttle_raw.startswith("0x"):
                try:
                    mask = int(throttle_raw, 16)
                    reasons = []
                    if mask & 0x1: reasons.append("Idle")
                    if mask & 0x2: reasons.append("App Clocks")
                    if mask & 0x4: reasons.append("Power Cap")
                    if mask & 0x8: reasons.append("HW Slowdown")
                    if mask & 0x10: reasons.append("SW Power Cap")
                    if mask & 0x20: reasons.append("Thermal")
                    throttle_str = ", ".join(reasons) if reasons else "None"
                except:
                    throttle_str = "Unknown"

            gpus.append(
                GpuInfo(
                    index=idx,
                    name=name,
                    util=util,
                    mem_used=mem_used,
                    mem_total=mem_total,
                    temp=temp,
                    power=power,
                    power_limit=power_limit,
                    fan=fan,
                    sm_clock_mhz=sm_clock,
                    throttle_reasons=throttle_str,
                )
            )

        # Second lightweight query for SM count (GPU cores) - best effort
        try:
            sm_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,multiprocessor_count",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if sm_result.returncode == 0:
                sm_reader = csv.reader(sm_result.stdout.strip().splitlines())
                sm_map = {}
                for sm_row in sm_reader:
                    if len(sm_row) >= 2:
                        sm_map[int(sm_row[0])] = int(sm_row[1]) if sm_row[1].strip().isdigit() else 0
                for g in gpus:
                    if g.index in sm_map:
                        g.sm_count = sm_map[g.index]
        except Exception:
            pass  # SM count is optional

        # Estimate CUDA cores using GPU name (works even when nvidia-smi doesn't expose SM count)
        for g in gpus:
            if g.cuda_cores == 0:
                g.cuda_cores = estimate_cuda_cores(g.name)

        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        return []


# Rough mapping of common NVIDIA GPU names → CUDA core counts.
# This is used as a fallback when nvidia-smi doesn't expose the info directly.
_CUDA_CORE_LOOKUP = {
    # RTX 30-series (Ampere)
    "rtx 3050": 2048,
    "rtx 3060": 3584,
    "rtx 3060 ti": 4864,
    "rtx 3070": 5888,
    "rtx 3070 ti": 6144,
    "rtx 3080": 8704,
    "rtx 3080 ti": 10240,
    "rtx 3090": 10496,
    "rtx 3090 ti": 10752,
    # RTX 40-series (Ada Lovelace)
    "rtx 4050": 2560,
    "rtx 4060": 3072,
    "rtx 4060 ti": 4352,
    "rtx 4070": 5888,
    "rtx 4070 ti": 7680,
    "rtx 4080": 9728,
    "rtx 4090": 16384,
    # Common older cards
    "rtx 2060": 1920,
    "rtx 2070": 2304,
    "rtx 2080": 2944,
    "gtx 1660": 1408,
    "gtx 1660 ti": 1536,
}


def estimate_cuda_cores(gpu_name: str) -> int:
    """Try to determine CUDA core count from the GPU name."""
    name = gpu_name.lower()
    for key, cores in _CUDA_CORE_LOOKUP.items():
        if key in name:
            return cores
    return 0


def detect_tpus() -> list[str]:
    """Detect available TPUs / ML accelerators on the system."""
    devices = []
    try:
        # Check lspci for TPU-like devices
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=2
        )
        for line in result.stdout.splitlines():
            lower = line.lower()
            if any(k in lower for k in ["tpu", "edge tpu", "coral", "google", "habana", "gaudi"]):
                devices.append(line.strip())
    except Exception:
        pass

    # Check for known device files (Google Coral)
    try:
        for dev in glob.glob("/dev/apex*"):
            devices.append(f"Device: {dev}")
    except Exception:
        pass

    # Deduplicate
    return list(dict.fromkeys(devices))


def get_tpu_processes() -> list[dict]:
    """Find processes that appear to be using TPU devices."""
    tpu_procs = []
    tpu_keywords = ["/dev/apex", "tpu", "edge_tpu"]

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            # Check open files for TPU device paths
            for f in proc.open_files():
                if any(kw in f.path.lower() for kw in tpu_keywords):
                    tpu_procs.append({
                        "pid": proc.pid,
                        "name": proc.name(),
                    })
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Dedup by pid
    seen = set()
    unique = []
    for p in tpu_procs:
        if p["pid"] not in seen:
            seen.add(p["pid"])
            unique.append(p)

    return unique


def get_system_memory() -> tuple[float, float]:
    """Return (used_gb, total_gb)."""
    vm = psutil.virtual_memory()
    return vm.used / (1024**3), vm.total / (1024**3)


def get_file_stats() -> dict[str, float]:
    """Return basic open file descriptor stats (Linux /proc/sys/fs/file-nr)."""
    try:
        with open("/proc/sys/fs/file-nr") as f:
            allocated, unused, max_files = map(int, f.read().strip().split())
        used = allocated - unused
        display_max = max_files if max_files < 10_000_000 else "unlimited"
        pct = (used / max_files * 100) if max_files > 0 and max_files < 10_000_000 else 0
        return {
            "used": used,
            "max": display_max,
            "pct": min(100.0, pct),
        }
    except Exception:
        return {"used": 0, "max": 0, "pct": 0}


def get_cpu_model() -> str:
    """Return a reasonably short CPU model name."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    # Remove marketing fluff
                    model = (model
                             .replace("(R)", "")
                             .replace("(TM)", "")
                             .replace(" CPU", "")
                             .replace(" Processor", "")
                             .strip())
                    # Try to shorten to something like "Core i7-7700"
                    if "@" in model:
                        model = model.split("@")[0].strip()
                    if len(model) > 32:
                        model = model[:29] + "..."
                    return model
    except Exception:
        pass
    import platform
    return platform.processor() or "Unknown CPU"


def get_enhanced_system_stats() -> dict:
    """Collect more useful system-wide stats for the SYSTEM panel."""
    stats = {}

    # Memory
    vm = psutil.virtual_memory()
    stats["ram_used"] = vm.used / (1024**3)
    stats["ram_total"] = vm.total / (1024**3)
    stats["ram_pct"] = vm.percent

    # Swap
    swap = psutil.swap_memory()
    stats["swap_used"] = swap.used / (1024**3)
    stats["swap_total"] = swap.total / (1024**3)
    stats["swap_pct"] = swap.percent

    # File descriptors
    fstats = get_file_stats()
    stats.update(fstats)

    # Disk usage (root)
    try:
        du = psutil.disk_usage("/")
        stats["disk_used"] = du.used / (1024**3)
        stats["disk_total"] = du.total / (1024**3)
        stats["disk_pct"] = du.percent
    except Exception:
        stats["disk_used"] = 0
        stats["disk_total"] = 0
        stats["disk_pct"] = 0

    # Detailed process counts (total + running, broken down by user vs system)
    try:
        user_total = 0
        user_running = 0
        system_total = 0
        system_running = 0
        total_running = 0

        for proc in psutil.process_iter(['uids', 'status']):
            try:
                uid = proc.info['uids'].real
                status = proc.info['status']

                is_running = status == psutil.STATUS_RUNNING

                if is_running:
                    total_running += 1

                if uid == 0:
                    system_total += 1
                    if is_running:
                        system_running += 1
                else:
                    user_total += 1
                    if is_running:
                        user_running += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        stats["process_total"] = user_total + system_total
        stats["process_running"] = total_running
        stats["user_total"] = user_total
        stats["user_running"] = user_running
        stats["system_total"] = system_total
        stats["system_running"] = system_running
    except Exception:
        stats["process_total"] = 0
        stats["process_running"] = 0
        stats["user_total"] = 0
        stats["user_running"] = 0
        stats["system_total"] = 0
        stats["system_running"] = 0

    # IO Wait (CPU time spent waiting on disk I/O)
    try:
        times = psutil.cpu_times_percent(interval=0.05)
        stats["iowait"] = getattr(times, "iowait", 0.0)
    except Exception:
        stats["iowait"] = 0.0

    return stats


def get_top_cpu_processes(n: int = 3) -> list[CpuProcess]:
    """Return the top N processes by CPU usage (with a short measurement window)."""
    # First prime the system counters
    psutil.cpu_percent(interval=0.0)

    candidates = []
    for proc in psutil.process_iter(['name', 'memory_info']):
        try:
            name = proc.name() or "unknown"
            mem = proc.memory_info().rss / (1024 * 1024)
            # Non-blocking read (will be 0 on first call per process object)
            cpu = proc.cpu_percent(interval=0.0)
            candidates.append(CpuProcess(name=name, cpu=cpu, mem_mib=mem))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    candidates.sort(key=lambda p: p.cpu, reverse=True)

    # If the top ones are all 0 (common on first frames), do a quick measurement pass
    if candidates and candidates[0].cpu < 0.1:
        for p in candidates[: max(n, 8)]:
            try:
                proc = psutil.Process()  # dummy to avoid lookup cost
                # We can't easily re-measure without the original object.
                # For better results in practice, the repeated calls across TUI frames work well.
                pass
            except Exception:
                pass

    return candidates[:n]


def _nice_gpu_process_name(pid: int, fallback: str) -> str:
    """Try to get a more useful process name using psutil (e.g. script name for python)."""
    try:
        p = psutil.Process(pid)
        cmd = p.cmdline()
        if cmd:
            exe = os.path.basename(cmd[0])
            if len(cmd) > 1 and ("python" in exe or "python3" in exe):
                script = os.path.basename(cmd[1])
                if script and not script.startswith("-"):
                    return f"{exe} ({script})"
            return exe
        return p.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return fallback


def get_gpu_processes(gpus: list[GpuInfo]) -> list[GpuProcess]:
    """Collect GPU processes using both query-compute-apps (memory) + pmon (real utilization)."""
    if not gpus:
        return []

    # Build maps
    uuid_to_index: dict[str, int] = {}
    index_to_total: dict[int, float] = {g.index: g.mem_total for g in gpus}

    try:
        uuid_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if uuid_result.returncode == 0:
            for line in uuid_result.stdout.strip().splitlines():
                if "," in line:
                    idx_str, uuid = [x.strip() for x in line.split(",", 1)]
                    uuid_to_index[uuid] = int(idx_str)
    except Exception:
        pass

    # Step 1: Get accurate memory usage + basic name
    mem_map: dict[tuple[int, int], float] = {}   # (gpu_index, pid) -> mem_mib
    fallback_names: dict[tuple[int, int], str] = {}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            reader = csv.reader(result.stdout.strip().splitlines())
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    pid = int(row[0])
                    name = row[1].strip()
                    mem_mib = float(row[2])
                    uuid = row[3].strip()
                except ValueError:
                    continue

                gpu_index = uuid_to_index.get(uuid, 0)
                key = (gpu_index, pid)
                mem_map[key] = mem_mib
                fallback_names[key] = name
    except Exception:
        pass

    # Step 2: Get live utilization from pmon (the good stuff)
    pmon_data: dict[tuple[int, int], tuple[float, float]] = {}  # (gpu, pid) -> (sm, mem_util)

    try:
        pmon = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            capture_output=True, text=True, timeout=3,
        )
        if pmon.returncode == 0:
            for line in pmon.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    gpu_idx = int(parts[0])
                    pid = int(parts[1])
                    sm = float(parts[3]) if parts[3].replace(".", "", 1).isdigit() else 0.0
                    memu = float(parts[4]) if parts[4].replace(".", "", 1).isdigit() else 0.0
                    pmon_data[(gpu_idx, pid)] = (max(0, min(100, sm)), max(0, min(100, memu)))
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass

    # Merge everything
    procs: list[GpuProcess] = []
    all_keys = set(mem_map.keys()) | set(pmon_data.keys())

    for key in all_keys:
        gpu_index, pid = key
        mem_mib = mem_map.get(key, 0.0)
        sm_util, mem_util = pmon_data.get(key, (0.0, 0.0))

        # If we only have pmon data (rare), still show it
        total = index_to_total.get(gpu_index, 8192)
        mem_pct = (mem_mib / total * 100.0) if total > 0 else 0.0

        fallback = fallback_names.get(key, f"pid:{pid}")
        name = _nice_gpu_process_name(pid, fallback)

        procs.append(GpuProcess(
            pid=pid,
            name=name,
            gpu_index=gpu_index,
            mem_mib=mem_mib,
            mem_pct=min(100.0, mem_pct),
            sm_util=sm_util,
            mem_util=mem_util,
        ))

    # Sort by combined activity (prefer actual utilization, fall back to memory)
    def sort_key(p: GpuProcess):
        activity = p.sm_util * 1.5 + p.mem_util
        return (activity, p.mem_mib)

    procs.sort(key=sort_key, reverse=True)
    return procs[:6]  # keep a few more so per-GPU filtering still has good candidates


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_header(cpu: CpuSnapshot, term_width: int) -> Panel:
    """Top info bar: hostname, uptime, load averages."""
    hostname = os.uname().nodename
    up = cpu.uptime
    hours = up.seconds // 3600
    minutes = (up.seconds % 3600) // 60
    seconds = up.seconds % 60
    up_str = f"{up.days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"

    l1, l2, l3 = cpu.load1, cpu.load5, cpu.load15
    cores = len(cpu.per_core) or psutil.cpu_count() or 1

    load_text = Text.assemble(
        ("LOAD ", "dim"),
        (f"{l1:.2f}", f"bold {load_color(l1, cores)}"),
        "  ",
        (f"{l2:.2f}", f"bold {load_color(l2, cores)}"),
        "  ",
        (f"{l3:.2f}", f"bold {load_color(l3, cores)}"),
    )

    title = Text("THERMINAL", style="bold cyan")
    time_str = datetime.now().strftime("%H:%M:%S")

    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=1)
    table.add_column(justify="center", ratio=1)
    table.add_column(justify="right", ratio=1)

    table.add_row(
        Text.assemble(
            (hostname, "bold white"),
            ("  •  ", "dim"),
            ("UP ", "bold dim"),
            (up_str, "bold cyan"),
        ),
        title,
        Text.assemble(
            (time_str, "dim"),
            ("  •  ", "dim"),
            load_text,
        ),
    )

    return Panel(
        table,
        box=box.HEAVY,
        border_style="cyan",
        padding=(0, 1),
    )


def render_cpu_panel(cpu: CpuSnapshot, hottest_core_history: list[int] | None = None) -> Panel:
    """CPU panel: load, per-core, temperatures, frequency."""
    cores = len(cpu.per_core)
    physical = psutil.cpu_count(logical=False) or cores // 2

    # Overall big bar
    overall_bar = make_bar(cpu.overall, width=17, color=util_color(cpu.overall), label="TOTAL")

    # Per-core utilization for all 8 logical threads (correct for Hyper-Threading)
    core_table = Table.grid(padding=(0, 1))
    core_table.add_column(justify="right", style="dim", width=3)
    core_table.add_column()
    core_table.add_column(justify="right", style="dim", width=3)
    core_table.add_column()

    # Color the logical core labels (C0–C7) based on hottest physical core history over a large sampling window
    # (orange = moderately often hottest, red = very frequently the hottest over time)
    hot_freq = {}
    if hottest_core_history:
        counter = Counter(hottest_core_history)
        total = len(hottest_core_history)
        for phys_idx, count in counter.items():
            hot_freq[phys_idx] = count / total

    for i in range(0, cores, 2):
        phys_idx = i // 2
        freq = hot_freq.get(phys_idx, 0.0)
        if freq > 0.60:
            label_style = "bold red"
        elif freq > 0.30:
            label_style = "bold orange1"
        else:
            label_style = "dim"

        row = [Text(f"C{i}", style=label_style), small_bar(cpu.per_core[i])]

        if i + 1 < cores:
            phys_idx1 = (i + 1) // 2
            freq1 = hot_freq.get(phys_idx1, 0.0)
            if freq1 > 0.60:
                label_style1 = "bold red"
            elif freq1 > 0.30:
                label_style1 = "bold orange1"
            else:
                label_style1 = "dim"
            row.extend([Text(f"C{i+1}", style=label_style1), small_bar(cpu.per_core[i + 1])])
        else:
            row.extend(["", ""])

        core_table.add_row(*row)

    # Physical core temperatures (only 4 real sensors) + long-term favored %.
    # The % column directly shows which physical cores run hottest most often.
    phys_temp_lines = []
    if cpu.temp_cores:
        phys_temp_table = Table.grid(padding=(0, 2))
        phys_temp_table.add_column(style="dim", width=8)
        phys_temp_table.add_column(justify="right", width=8)
        phys_temp_table.add_column(justify="right", style="dim", width=5)

        for phys_idx, (label, temp) in enumerate(cpu.temp_cores):
            c = temp_color(temp)
            pct = hot_freq.get(phys_idx, 0.0) * 100
            pct_text = Text(f"{pct:.0f}%", style=("bold red" if pct > 50 else ("yellow" if pct > 28 else "dim"))) if pct >= 1 else Text("")
            phys_temp_table.add_row(
                label,
                Text(f"{temp:5.1f}°C", style=f"bold {c}"),
                pct_text
            )

        phys_temp_lines = [
            Text(""),
            Text("Physical Core Temps (4 sensors) + % time as hottest:", style="bold dim"),
            phys_temp_table
        ]

    # PKG temperature + delta (useful for detecting poor thermal paste / uneven contact)
    pkg_temp_text = Text("")
    if cpu.temp_package is not None:
        temp_delta = 0.0
        delta_color = "green"

        if len(cpu.temp_cores) > 1:
            core_temps = [t for _, t in cpu.temp_cores]
            temp_delta = max(core_temps) - min(core_temps)
            delta_color = "red" if temp_delta > 10 else ("yellow" if temp_delta > 6 else "green")

        c_pkg = delta_color if temp_delta > 6 else temp_color(cpu.temp_package)

        pkg_line = Text.assemble(
            ("PKG ", "dim"),
            (f"{cpu.temp_package:5.1f}°C", f"bold {c_pkg}"),
        )

        if temp_delta > 0:
            delta_text = Text.assemble(
                ("   Δ ", "dim"),
                (f"{temp_delta:.1f}°C", f"bold {delta_color}"),
            )
            pkg_line = Text.assemble(pkg_line, delta_text)

        pkg_temp_text = pkg_line

    # Favored core distribution — extremely useful for spotting persistent thermal
    # bias, bad paste, or cooling issues on specific physical cores over time.
    favored_line = Text("")
    if hottest_core_history and hot_freq:
        parts: list[Text] = [Text("Favored (1000-sample): ", style="dim")]
        leader_idx = max(hot_freq, key=hot_freq.get) if hot_freq else 0
        for p in range(physical):
            pct = hot_freq.get(p, 0.0) * 100
            if pct < 0.5:
                continue
            col = "red" if pct > 55 else ("yellow" if pct > 30 else "cyan")
            is_leader = (p == leader_idx and pct > 20)
            style = f"bold {col}" if is_leader else "dim"
            parts.append(Text(f" C{p}:{pct:.0f}%", style=style))
        if len(parts) > 1:
            leader_label = ""
            if leader_idx < len(cpu.temp_cores):
                leader_label = cpu.temp_cores[leader_idx][0]
            parts.append(Text(f"  lead {leader_label}", style="bold red"))
            favored_line = Text.assemble(*parts)

    model = get_cpu_model()
    freq_ghz = cpu.freq_mhz / 1000

    first_line = Text.assemble(
        (model, "bold white"),
        ("   ", "dim"),
        (f"{physical}c / {cores}t", "cyan"),
        ("   ", "dim"),
        (f"{freq_ghz:.2f} GHz", "cyan"),
    )

    content = Group(
        first_line,
        Text(""),
        overall_bar,
        Text(""),
        pkg_temp_text,
        favored_line,
        Text(""),
        Text("CORES (4 physical / 8 logical threads)", style="bold dim"),
        core_table,
        *phys_temp_lines,
    )

    return Panel(
        content,
        title="[bold cyan]CPU[/]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_system_panel() -> Panel:
    """SYSTEM panel with more useful stats."""
    s = get_enhanced_system_stats()

    # RAM
    ram_bar = make_bar(s["ram_pct"], width=16, color=util_color(s["ram_pct"]), label="RAM")

    # Swap (only show if present)
    swap_lines = []
    if s["swap_total"] > 0.1:
        swap_bar = make_bar(s["swap_pct"], width=16, color=util_color(s["swap_pct"]), label="SWAP")
        swap_lines = [
            swap_bar,
            Text(f"{s['swap_used']:.1f} / {s['swap_total']:.1f} GB", style="dim"),
            Text(""),
        ]

    # Disk + IO Wait
    disk_bar = make_bar(s["disk_pct"], width=16, color=util_color(s["disk_pct"]), label="DISK")

    iowait_bar = make_bar(s["iowait"], width=16, color=util_color(s["iowait"]), label="IOWAIT")

    # Files - nicer display when max is "unlimited"
    if isinstance(s["max"], str) or s["max"] > 1_000_000:
        files_text = f"{int(s['used']):,} open fds"
    else:
        files_text = f"{int(s['used']):,} / {int(s['max']):,} fds"

    file_bar = make_bar(s["pct"], width=16, color=util_color(s["pct"]), label="FILES")

    content = Group(
        ram_bar,
        Text(f"{s['ram_used']:.1f} / {s['ram_total']:.1f} GB", style="dim"),
        *swap_lines,
        disk_bar,
        Text(f"{s['disk_used']:.1f} / {s['disk_total']:.1f} GB", style="dim"),
        iowait_bar,
        Text(f"{s['iowait']:.1f}% CPU waiting on I/O", style="dim"),
        file_bar,
        Text(files_text, style="dim"),
    )

    # Add TPU info if present (simple presence)
    tpu_devices = detect_tpus()
    if tpu_devices:
        content = Group(
            *content.renderables,
            Text(""),
            Text(f"TPU: {len(tpu_devices)} device(s) detected", style="magenta"),
        )

    # Helpful admin / operator info (kernel, sessions, last reboot)
    try:
        host_info: list[RenderableType] = []
        u = os.uname()
        host_info.append(Text(f"Kernel: {u.release}", style="dim"))
        try:
            n_users = len(psutil.users())
            host_info.append(Text(f"Users: {n_users} logged in", style="dim"))
        except Exception:
            pass
        boot = datetime.fromtimestamp(psutil.boot_time())
        host_info.append(Text(f"Booted: {boot.strftime('%Y-%m-%d %H:%M')}", style="dim"))

        if host_info:
            content = Group(
                *content.renderables,
                Text(""),
                Text("HOST", style="bold dim"),
                *host_info,
            )
    except Exception:
        pass

    return Panel(
        content,
        title="[bold magenta]SYSTEM[/]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_processes_panel(
    cpu_procs: list[CpuProcess], 
    gpu_procs: list[GpuProcess], 
    tpu_procs: list[dict]
) -> Panel:
    """Dedicated PROCESSES panel with Top 3 for CPU and GPU (plus TPU if present)."""
    lines: list[RenderableType] = []

    # One-line process summary at the top (moved from SYSTEM panel)
    s = get_enhanced_system_stats()
    proc_summary = Text.assemble(
        ("PROCS ", "dim"),
        ("A:", "dim"), (f"{s['process_total']}|{s['process_running']}", "cyan"),
        (" S:", "dim"), (f"{s['system_total']}|{s['system_running']}", "cyan"),
        (" U:", "dim"), (f"{s['user_total']}|{s['user_running']}", "cyan"),
    )
    lines.append(proc_summary)
    lines.append(Text(""))

    # Top 3 CPU processes
    if cpu_procs:
        lines.append(Text("Top 3 CPU", style="bold yellow"))
        t = Table.grid(padding=(0, 0))
        t.add_column(style="white", width=14)
        t.add_column(style="yellow", width=7)
        t.add_column(style="cyan", width=6, justify="right")
        t.add_column(width=9)

        for p in cpu_procs[:3]:
            mem_str = f"{p.mem_mib / 1024:.1f}G" if p.mem_mib >= 1024 else f"{p.mem_mib:.0f}M"
            cpu_str = f"{p.cpu:5.1f}%"
            bar = make_memory_bar(min(p.cpu, 100), width=9)
            t.add_row(p.name[:13], cpu_str, mem_str, bar)
        lines.append(t)

    # Top 3 GPU processes
    if gpu_procs:
        if cpu_procs:
            lines.append(Text(""))
        lines.append(Text("Top 3 GPU", style="bold green"))
        t = Table.grid(padding=(0, 0))
        t.add_column(style="white", width=14)
        t.add_column(style="cyan", width=6, justify="right")
        t.add_column(style="yellow", width=7)
        t.add_column(width=9)

        for p in gpu_procs[:3]:
            mem_str = f"{p.mem_mib / 1024:.1f}G" if p.mem_mib >= 1024 else f"{p.mem_mib:.0f}M"
            util_str = f"sm{int(p.sm_util)}%" if p.sm_util > 0.5 else ""
            bar = make_memory_bar(p.sm_util if p.sm_util > 0.1 else p.mem_pct, width=9)
            t.add_row(p.name[:13], mem_str, util_str, bar)
        lines.append(t)

    # TPU processes (if any detected)
    if tpu_procs:
        if cpu_procs or gpu_procs:
            lines.append(Text(""))
        lines.append(Text("Top TPU", style="bold magenta"))
        t = Table.grid(padding=(0, 0))
        t.add_column(style="white", width=14)
        t.add_column(style="cyan", width=6, justify="right")
        t.add_column(width=12)

        for p in tpu_procs[:3]:
            t.add_row(p["name"][:13], str(p["pid"]), "using TPU")
        lines.append(t)

    if not lines:
        content = Text("No significant processes detected", style="dim")
    else:
        content = Group(*lines)

    return Panel(
        content,
        title="[bold green]PROCESSES[/]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_gpu_panel(gpus: list[GpuInfo], procs: list[GpuProcess]) -> Panel:
    """Right panel: GPU cards (processes now shown in the dedicated PROCESSES panel)."""
    if not gpus:
        return Panel(
            Text("No NVIDIA GPU detected (nvidia-smi unavailable)", style="dim"),
            title="[bold]GPU[/]",
            border_style="dim",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    sections: list[RenderableType] = []

    for i, g in enumerate(gpus):
        if i > 0:
            sections.append(Text(""))

        # First line: GPU name + key specs (CUDA cores + SMs)
        name_parts = [Text(g.name, style="bold white")]

        if g.cuda_cores > 0 and g.sm_count > 0:
            name_parts.extend([
                Text("  •  ", style="dim"),
                Text(f"{g.cuda_cores}c", style="cyan"),
                Text(f" ({g.sm_count} SMs)", style="dim"),
            ])
        elif g.cuda_cores > 0:
            name_parts.extend([
                Text("  •  ", style="dim"),
                Text(f"{g.cuda_cores}c", style="cyan"),
            ])
        elif g.sm_count > 0:
            name_parts.extend([
                Text("  •  ", style="dim"),
                Text(f"{g.sm_count} SMs", style="cyan"),
            ])

        name_line = Text.assemble(*name_parts)

        util_bar = make_bar(g.util, width=17, color=util_color(g.util), label="UTIL")
        mem_pct = (g.mem_used / g.mem_total * 100) if g.mem_total > 0 else 0
        mem_bar = make_bar(mem_pct, width=17, color=util_color(mem_pct), label="VRAM")

        tcol = temp_color(g.temp, is_gpu=True)
        temp_text = Text.assemble(
            ("TEMP  ", "dim"),
            (f"{g.temp:5.1f}°C", f"bold {tcol}"),
        )

        pcol = power_color(g.power, g.power_limit)
        power_text = Text.assemble(
            ("PWR   ", "dim"),
            (f"{g.power:6.1f}W", f"bold {pcol}"),
        )
        if g.power_limit > 0:
            power_text.append(Text(f" / {g.power_limit:.0f}W", style="dim"))

        fan_text = Text("")
        if g.fan >= 0:
            fan_text = Text.assemble(
                ("FAN   ", "dim"),
                (f"{g.fan:5.1f}%", "cyan"),
            )

        mem_detail = Text.assemble(
            (f"{g.mem_used / 1024:.1f} / {g.mem_total / 1024:.1f} GB", "dim"),
        )

        # GPU Performance / Clock info
        perf_text = Text("")
        if g.sm_clock_mhz > 0:
            clock_str = f"{g.sm_clock_mhz} MHz"
            if g.throttle_reasons and g.throttle_reasons != "None":
                clock_str += f" (throttled: {g.throttle_reasons})"
            perf_text = Text.assemble(
                ("CLOCK ", "dim"),
                (clock_str, "cyan"),
            )

        gpu_block = Group(
            name_line,
            Text(""),
            util_bar,
            mem_bar,
            mem_detail,
            Text(""),
            temp_text,
            power_text,
            perf_text,
            fan_text,
        )
        sections.append(gpu_block)

    title = "[bold green]GPU[/]" if len(gpus) == 1 else f"[bold green]GPU ×{len(gpus)}[/]"
    return Panel(
        Group(*sections),
        title=title,
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_footer(interval: float, gpus: list[GpuInfo]) -> RenderableType:
    gpu_status = "NVIDIA" if gpus else "CPU only"
    help_text = Text.assemble(
        ("q", "bold yellow"),
        " quit   ",
        ("+/-", "bold yellow"),
        " refresh rate   ",
        (f"{interval:.1f}s", "cyan"),
        "   •   ",
        (gpu_status, "dim"),
    )
    return help_text


def build_layout(
    cpu: CpuSnapshot,
    gpus: list[GpuInfo],
    gpu_procs: list[GpuProcess],
    cpu_procs: list[CpuProcess],
    tpu_procs: list[dict],
    hottest_core_history: list[int],
    interval: float,
) -> Layout:
    term_width = shutil.get_terminal_size((100, 40)).columns

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=1),
    )

    layout["header"].update(render_header(cpu, term_width))

    # Create named sub-layouts for the 2x2
    cpu_l = Layout(name="cpu")
    gpu_l = Layout(name="gpu")
    system_l = Layout(name="system")
    proc_l = Layout(name="processes")

    top = Layout(name="top", ratio=5)
    top.split_row(cpu_l, gpu_l)

    bottom = Layout(name="bottom", ratio=5)
    bottom.split_row(system_l, proc_l)

    layout["body"].split_column(top, bottom)

    layout["cpu"].update(render_cpu_panel(cpu, hottest_core_history))
    layout["gpu"].update(render_gpu_panel(gpus, gpu_procs))
    layout["system"].update(render_system_panel())
    layout["processes"].update(render_processes_panel(cpu_procs, gpu_procs, tpu_procs))

    layout["footer"].update(render_footer(interval, gpus))

    return layout


# ──────────────────────────────────────────────────────────────────────────────
# Keyboard handling (non-blocking)
# ──────────────────────────────────────────────────────────────────────────────

class KeyReader:
    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *args):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self) -> str | None:
        if not sys.stdin.isatty():
            return None
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            ch = sys.stdin.read(1)
            return ch
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run(interval: float = 1.0):
    key_reader = KeyReader()

    tpu_devices = detect_tpus()

    # Track which physical core has been the hottest recently (for paste quality diagnostics)
    # Using a large window so the label coloring reflects long-term behavior, not short-term noise.
    hottest_history: deque[int] = deque(maxlen=_HOTTEST_HISTORY_WINDOW)

    with Live(
        build_layout(get_cpu_snapshot(), parse_nvidia_smi(), [], [], [], [], interval),
        console=console,
        screen=True,
        refresh_per_second=8,
        transient=False,
    ) as live:
        with key_reader:
            last_update = time.monotonic()

            while True:
                key = key_reader.get_key()
                if key in ("q", "Q", "\x03"):  # q or Ctrl+C
                    break
                if key in ("+", "=", "k"):
                    interval = max(0.3, interval - 0.2)
                if key in ("-", "_", "j"):
                    interval = min(5.0, interval + 0.2)
                if key in ("r", "R"):
                    last_update = 0  # force immediate refresh

                now = time.monotonic()
                if now - last_update >= interval:
                    cpu = get_cpu_snapshot()
                    gpus = parse_nvidia_smi()
                    gpu_procs = get_gpu_processes(gpus)
                    cpu_procs = get_top_cpu_processes(3)
                    tpu_procs = get_tpu_processes()

                    # Record the hottest core this frame
                    if cpu.temp_cores:
                        core_temps = [t for _, t in cpu.temp_cores]
                        hottest_idx = core_temps.index(max(core_temps))
                        hottest_history.append(hottest_idx)

                    live.update(build_layout(cpu, gpus, gpu_procs, cpu_procs, tpu_procs, list(hottest_history), interval))
                    last_update = now

                time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(
        description="Therminal — live GPU + CPU load, temperature, and system dashboard"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    try:
        run(max(0.3, min(args.interval, 5.0)))
    except KeyboardInterrupt:
        pass
    finally:
        console.show_cursor()
        console.print("\n[dim]Therminal exited.[/]")


if __name__ == "__main__":
    main()
