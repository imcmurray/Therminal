#!/usr/bin/env python3
"""
Therminal — Professional System Monitoring Dashboard

A beautiful, production-quality terminal UI for monitoring CPU, GPU, system resources,
and processes with smart alerting, excellent visual hierarchy, and clean design.

Version: 1.0.0
"""

__version__ = "1.0.0"

# Features:
# - Clean Rich-based panels with professional styling
# - Smart color coding (green/yellow/red)
# - Automatic issue detection and dedicated Alerts strip
# - Favored core analysis with long-term history
# - Smooth temperature readings (300-sample averaging)
# - Responsive layout using Rich Layout
# - Keyboard controls (q to quit, +/- for refresh rate)
#
# Usage:
#     python3 therminal.py
#     python3 therminal.py --interval 0.5
#     python3 therminal.py --debug
#     python3 therminal.py --version


import argparse
import csv
import glob
import os
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import psutil
from rich import box
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# =============================================================================
# CONFIGURATION & THRESHOLDS
# =============================================================================

# Temperature smoothing (reduces flicker while remaining responsive)
TEMP_SMOOTHING_WINDOW = 300          # ~5 minutes at 1s refresh
HOTTEST_HISTORY_WINDOW = 1000        # ~17 minutes for favored core analysis

# Alert thresholds (tunable)
CPU_TEMP_WARN = 80.0
CPU_TEMP_CRIT = 92.0
GPU_TEMP_WARN = 75.0
GPU_TEMP_CRIT = 85.0
RAM_WARN = 80.0
RAM_CRIT = 90.0
DISK_WARN = 85.0
DISK_CRIT = 95.0
LOAD_WARN = 1.0                      # per core
LOAD_CRIT = 2.0                      # per core
GPU_POWER_WARN_RATIO = 0.85
GPU_THROTTLE_ALERT = True

# Refresh defaults
DEFAULT_INTERVAL = 1.0
MIN_INTERVAL = 0.3
MAX_INTERVAL = 5.0

console = Console()


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GpuInfo:
    index: int
    name: str
    util: float
    mem_used: float
    mem_total: float
    temp: float
    power: float
    power_limit: float
    fan: float
    sm_clock_mhz: int = 0
    throttle_reasons: str = ""
    sm_count: int = 0
    cuda_cores: int = 0


@dataclass
class GpuProcess:
    name: str
    mem_mib: float
    sm_util: float = 0.0


@dataclass
class CpuProcess:
    name: str
    cpu: float
    mem_mib: float


@dataclass
class CpuSnapshot:
    overall: float
    per_core: list[float]
    freq_mhz: float
    temp_package: Optional[float]
    temp_cores: list[tuple[str, float]]
    load1: float
    load5: float
    load15: float
    uptime: timedelta


@dataclass
class SystemStats:
    ram_used: float
    ram_total: float
    ram_pct: float
    swap_used: float
    swap_total: float
    swap_pct: float
    disk_used: float
    disk_total: float
    disk_pct: float
    iowait: float
    open_fds: int
    fds_max: Any
    process_total: int
    process_running: int
    user_total: int
    user_running: int
    system_total: int
    system_running: int


# =============================================================================
# GLOBAL STATE (smoothing + history)
# =============================================================================

_package_temp_history: deque[float] = deque(maxlen=TEMP_SMOOTHING_WINDOW)
_core_temp_histories: dict[str, deque[float]] = {}
_hottest_history: deque[int] = deque(maxlen=HOTTEST_HISTORY_WINDOW)


# =============================================================================
# COLOR SYSTEM (professional & consistent)
# =============================================================================

def status_color(value: float, warn: float, crit: float, higher_is_worse: bool = True) -> str:
    """Return green / yellow / red based on thresholds."""
    if higher_is_worse:
        if value >= crit:
            return "red"
        if value >= warn:
            return "yellow"
        return "green"
    else:
        if value <= crit:   # lower is worse (e.g. fan speed)
            return "red"
        if value <= warn:
            return "yellow"
        return "green"


def temp_color(temp: float, is_gpu: bool = False) -> str:
    if is_gpu:
        return status_color(temp, GPU_TEMP_WARN, GPU_TEMP_CRIT)
    return status_color(temp, CPU_TEMP_WARN, CPU_TEMP_CRIT)


def util_color(pct: float) -> str:
    return status_color(pct, 70.0, 90.0)


def load_color(load: float, cores: int) -> str:
    ratio = load / max(1, cores)
    return status_color(ratio, LOAD_WARN, LOAD_CRIT)


def power_color(current: float, limit: float) -> str:
    if limit <= 0:
        return "cyan"
    ratio = current / limit
    return status_color(ratio, 0.75, GPU_POWER_WARN_RATIO)


# =============================================================================
# BEAUTIFUL BAR RENDERING
# =============================================================================

def make_bar(value: float, width: int = 18, color: Optional[str] = None, label: str = "") -> Text:
    """Premium unicode progress bar."""
    clamped = max(0.0, min(100.0, value))
    filled = int(clamped / 100 * width)

    if color is None:
        color = util_color(clamped)

    bar = Text()
    if filled > 0:
        bar.append("█" * filled, style=color)
    if width - filled > 0:
        bar.append("░" * (width - filled), style="dim")

    pct = Text(f" {clamped:5.1f}%", style=f"bold {color}")

    if label:
        return Text.assemble(Text(f"{label} ", style="bold dim"), bar, pct)
    return Text.assemble(bar, pct)


def mini_bar(value: float, width: int = 8) -> Text:
    """Compact bar for process lists."""
    clamped = max(0.0, min(100.0, value))
    filled = int(clamped / 100 * width)
    color = util_color(clamped)
    return Text("█" * filled + "░" * (width - filled), style=color)


# =============================================================================
# DATA COLLECTION
# =============================================================================

def get_uptime() -> timedelta:
    with open("/proc/uptime") as f:
        seconds = float(f.readline().split()[0])
    return timedelta(seconds=int(seconds))


def get_loadavg() -> tuple[float, float, float]:
    with open("/proc/loadavg") as f:
        parts = f.readline().split()
    return float(parts[0]), float(parts[1]), float(parts[2])


def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    model = (model
                             .replace("(R)", "")
                             .replace("(TM)", "")
                             .replace(" CPU", "")
                             .replace(" Processor", "")
                             .strip())
                    if "@" in model:
                        model = model.split("@")[0].strip()
                    if len(model) > 28:
                        model = model[:25] + "..."
                    return model
    except Exception:
        pass
    return "Unknown CPU"


def get_cpu_snapshot() -> CpuSnapshot:
    """Collect CPU data with heavy smoothing and favored-core history."""
    overall = psutil.cpu_percent(interval=0.12)
    per_core = psutil.cpu_percent(percpu=True, interval=0.0)

    freq = psutil.cpu_freq()
    freq_mhz = freq.current if freq else 0.0

    # Temperatures with smoothing
    temps = psutil.sensors_temperatures()
    raw_pkg: Optional[float] = None
    raw_cores: list[tuple[str, float]] = []

    if "coretemp" in temps:
        for entry in temps["coretemp"]:
            label_lower = (entry.label or "").lower()
            if "package" in label_lower or entry.label == "Package id 0":
                raw_pkg = entry.current
            elif "core" in label_lower:
                lbl = entry.label or f"Core {len(raw_cores)}"
                raw_cores.append((lbl, entry.current))

    # Update histories
    if raw_pkg is not None:
        _package_temp_history.append(raw_pkg)
    for label, t in raw_cores:
        if label not in _core_temp_histories:
            _core_temp_histories[label] = deque(maxlen=TEMP_SMOOTHING_WINDOW)
        _core_temp_histories[label].append(t)

    # Smoothed values
    pkg = sum(_package_temp_history) / len(_package_temp_history) if _package_temp_history else None
    smoothed_cores = []
    for label, _ in raw_cores:
        hist = _core_temp_histories.get(label)
        if hist:
            smoothed_cores.append((label, sum(hist) / len(hist)))

    # Record hottest physical core (for favored core analysis)
    if smoothed_cores:
        core_temps = [t for _, t in smoothed_cores]
        hottest_idx = core_temps.index(max(core_temps))
        _hottest_history.append(hottest_idx)

    load1, load5, load15 = get_loadavg()
    uptime = get_uptime()

    return CpuSnapshot(
        overall=overall,
        per_core=per_core,
        freq_mhz=freq_mhz,
        temp_package=pkg,
        temp_cores=smoothed_cores,
        load1=load1, load5=load5, load15=load15,
        uptime=uptime,
    )


def parse_nvidia_smi(debug: bool = False) -> list[GpuInfo]:
    """GPU data with throttle detection + CUDA core / SM count."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit,fan.speed,"
             "clocks.current.sm,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return []

        gpus = []
        for row in csv.reader(result.stdout.strip().splitlines()):
            if len(row) < 11:
                continue
            g = GpuInfo(
                index=int(row[0]),
                name=row[1].strip(),
                util=float(row[2]),
                mem_used=float(row[3]),
                mem_total=float(row[4]),
                temp=float(row[5]),
                power=float(row[6]) if row[6].strip() else 0.0,
                power_limit=float(row[7]) if row[7].strip() else 0.0,
                fan=float(row[8]) if row[8].strip() and row[8].strip() != "-1" else -1.0,
                sm_clock_mhz=int(row[9]) if row[9].strip().isdigit() else 0,
            )
            # Throttle reasons
            raw = row[10].strip() if len(row) > 10 else ""
            if raw.startswith("0x"):
                mask = int(raw, 16)
                reasons = []
                if mask & 0x2: reasons.append("App Clocks")
                if mask & 0x4: reasons.append("Power Cap")
                if mask & 0x8: reasons.append("HW Slowdown")
                if mask & 0x20: reasons.append("Thermal")
                g.throttle_reasons = ", ".join(reasons) if reasons else "None"

            gpus.append(g)

        # Second lightweight query for SM count (best effort)
        try:
            sm_result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,multiprocessor_count",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if sm_result.returncode == 0:
                for sm_row in csv.reader(sm_result.stdout.strip().splitlines()):
                    if len(sm_row) >= 2:
                        idx = int(sm_row[0])
                        sm_count = int(sm_row[1]) if sm_row[1].strip().isdigit() else 0
                        for g in gpus:
                            if g.index == idx:
                                g.sm_count = sm_count
        except Exception:
            pass

        # Estimate CUDA cores from name as fallback
        for g in gpus:
            if g.cuda_cores == 0:
                g.cuda_cores = estimate_cuda_cores(g.name)

        return gpus
    except Exception as e:
        if debug:
            print(f"[DEBUG] nvidia-smi query failed: {e}", file=sys.stderr)
        return []


# Rough mapping of common NVIDIA GPU names → CUDA core counts.
# Used as a reliable fallback when nvidia-smi doesn't expose multiprocessor_count.
_CUDA_CORE_LOOKUP = {
    # RTX 30-series (Ampere)
    "rtx 3050": 2048, "rtx 3060": 3584, "rtx 3060 ti": 4864,
    "rtx 3070": 5888, "rtx 3070 ti": 6144, "rtx 3080": 8704,
    "rtx 3080 ti": 10240, "rtx 3090": 10496, "rtx 3090 ti": 10752,
    # RTX 40-series (Ada Lovelace)
    "rtx 4050": 2560, "rtx 4060": 3072, "rtx 4060 ti": 4352,
    "rtx 4070": 5888, "rtx 4070 ti": 7680, "rtx 4080": 9728,
    "rtx 4090": 16384,
    # Older cards
    "rtx 2060": 1920, "rtx 2070": 2304, "rtx 2080": 2944,
    "gtx 1660": 1408, "gtx 1660 ti": 1536,
}


def estimate_cuda_cores(gpu_name: str) -> int:
    """Best-effort CUDA core count from GPU name."""
    name = gpu_name.lower()
    for key, cores in _CUDA_CORE_LOOKUP.items():
        if key in name:
            return cores
    return 0


def get_gpu_processes(gpus: list[GpuInfo], debug: bool = False) -> list[GpuProcess]:
    """
    Collect top GPU processes with real SM utilization (from pmon) + accurate VRAM usage.
    This is the improved version that merges two nvidia-smi queries.
    """
    if not gpus:
        return []

    procs_by_pid: dict[int, GpuProcess] = {}

    try:
        # 1. pmon gives us actual SM utilization (the important part)
        pmon = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            capture_output=True, text=True, timeout=2
        )
        for line in pmon.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    pid = int(parts[1])
                    sm_util = float(parts[4])
                    if sm_util > 0.5:  # only interesting processes
                        if pid not in procs_by_pid:
                            procs_by_pid[pid] = GpuProcess(name="unknown", mem_mib=0.0, sm_util=0.0)
                        procs_by_pid[pid].sm_util = max(procs_by_pid[pid].sm_util, sm_util)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        if debug:
            print(f"[DEBUG] GPU pmon query failed: {e}", file=sys.stderr)

    try:
        # 2. query-compute-apps gives accurate per-process memory
        mem_result = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        for row in csv.reader(mem_result.stdout.strip().splitlines()):
            if len(row) >= 2:
                try:
                    pid = int(row[0])
                    mem = float(row[1])
                    if pid in procs_by_pid:
                        procs_by_pid[pid].mem_mib = mem
                    elif mem > 50:  # show memory hogs even with low SM
                        procs_by_pid[pid] = GpuProcess(name="unknown", mem_mib=mem, sm_util=0.0)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        if debug:
            print(f"[DEBUG] GPU compute-apps query failed: {e}", file=sys.stderr)

    # Enrich names using psutil (best effort)
    for pid, proc in list(procs_by_pid.items()):
        try:
            p = psutil.Process(pid)
            cmd = p.cmdline()
            if cmd:
                exe = os.path.basename(cmd[0])
                if len(cmd) > 1 and ("python" in exe or "python3" in exe):
                    script = os.path.basename(cmd[1])
                    if script and not script.startswith("-"):
                        proc.name = f"{exe} ({script})"[:20]
                        continue
                proc.name = exe[:20]
            else:
                proc.name = p.name()[:20]
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            proc.name = f"pid:{pid}"

    # Sort by SM utilization (primary), then memory
    result = sorted(procs_by_pid.values(), key=lambda p: (p.sm_util, p.mem_mib), reverse=True)
    return result[:5]


def get_top_cpu_processes(n: int = 5, debug: bool = False) -> list[CpuProcess]:
    psutil.cpu_percent(interval=0.0)
    candidates = []
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = proc.info["name"] or "unknown"
            mem = proc.info["memory_info"].rss / (1024 * 1024)
            cpu = proc.cpu_percent(interval=0.0)
            candidates.append(CpuProcess(name=name[:18], cpu=cpu, mem_mib=mem))
        except Exception as e:
            if debug:
                print(f"[DEBUG] Failed to inspect CPU process: {e}", file=sys.stderr)
            continue
    candidates.sort(key=lambda p: p.cpu, reverse=True)
    return candidates[:n]


def get_system_stats(debug: bool = False) -> SystemStats:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    du = psutil.disk_usage("/")
    disk_pct = du.percent

    times = psutil.cpu_times_percent(interval=0.05)
    iowait = getattr(times, "iowait", 0.0)

    # File descriptors
    try:
        with open("/proc/sys/fs/file-nr") as f:
            alloc, unused, maxf = map(int, f.read().strip().split())
        open_fds = alloc - unused
        fds_max = "unlimited" if maxf > 10_000_000 else maxf
    except Exception as e:
        if debug:
            print(f"[DEBUG] Failed to read file descriptor stats: {e}", file=sys.stderr)
        open_fds, fds_max = 0, 0

    # Process breakdown
    user_t = user_r = sys_t = sys_r = total_r = 0
    for p in psutil.process_iter(["uids", "status"]):
        try:
            uid = p.info["uids"].real
            running = p.info["status"] == psutil.STATUS_RUNNING
            if running:
                total_r += 1
            if uid == 0:
                sys_t += 1
                if running: sys_r += 1
            else:
                user_t += 1
                if running: user_r += 1
        except Exception:
            continue

    return SystemStats(
        ram_used=vm.used / (1024**3),
        ram_total=vm.total / (1024**3),
        ram_pct=vm.percent,
        swap_used=swap.used / (1024**3),
        swap_total=swap.total / (1024**3),
        swap_pct=swap.percent,
        disk_used=du.used / (1024**3),
        disk_total=du.total / (1024**3),
        disk_pct=disk_pct,
        iowait=iowait,
        open_fds=open_fds,
        fds_max=fds_max,
        process_total=user_t + sys_t,
        process_running=total_r,
        user_total=user_t,
        user_running=user_r,
        system_total=sys_t,
        system_running=sys_r,
    )


def detect_issues(cpu: CpuSnapshot, gpus: list[GpuInfo], sysstats: SystemStats) -> list[str]:
    """Return list of current problems."""
    issues = []

    # CPU
    if cpu.temp_package and cpu.temp_package >= CPU_TEMP_CRIT:
        issues.append(f"CPU package critical ({cpu.temp_package:.0f}°C)")
    elif cpu.temp_package and cpu.temp_package >= CPU_TEMP_WARN:
        issues.append(f"CPU package warm ({cpu.temp_package:.0f}°C)")

    if cpu.temp_cores:
        hottest = max(t for _, t in cpu.temp_cores)
        if hottest >= CPU_TEMP_CRIT:
            issues.append(f"Hottest core critical ({hottest:.0f}°C)")

    cores = len(cpu.per_core) or psutil.cpu_count() or 4
    if cpu.load1 > cores * LOAD_CRIT:
        issues.append(f"Very high load ({cpu.load1:.1f})")
    elif cpu.load1 > cores * LOAD_WARN:
        issues.append(f"Elevated load ({cpu.load1:.1f})")

    # RAM / Disk
    if sysstats.ram_pct >= RAM_CRIT:
        issues.append(f"RAM critical ({sysstats.ram_pct:.0f}%)")
    elif sysstats.ram_pct >= RAM_WARN:
        issues.append(f"RAM high ({sysstats.ram_pct:.0f}%)")

    if sysstats.disk_pct >= DISK_CRIT:
        issues.append(f"Disk critical ({sysstats.disk_pct:.0f}%)")
    elif sysstats.disk_pct >= DISK_WARN:
        issues.append(f"Disk high ({sysstats.disk_pct:.0f}%)")

    # GPU
    for g in gpus:
        if g.temp >= GPU_TEMP_CRIT:
            issues.append(f"GPU {g.index} critical temp ({g.temp:.0f}°C)")
        elif g.temp >= GPU_TEMP_WARN:
            issues.append(f"GPU {g.index} warm ({g.temp:.0f}°C)")

        if g.power_limit > 0 and g.power / g.power_limit > GPU_POWER_WARN_RATIO:
            issues.append(f"GPU {g.index} high power draw")

        if g.throttle_reasons and g.throttle_reasons not in ("", "None"):
            issues.append(f"GPU {g.index} throttled ({g.throttle_reasons})")

    return issues[:6]  # cap


# =============================================================================
# RENDERING
# =============================================================================

def render_header(cpu: CpuSnapshot, term_width: int) -> Panel:
    """Top status bar."""
    hostname = os.uname().nodename
    up = cpu.uptime
    days = up.days
    hours = up.seconds // 3600
    minutes = (up.seconds % 3600) // 60
    seconds = up.seconds % 60

    if days > 0:
        up_str = f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
    else:
        up_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    load_text = Text.assemble(
        ("LOAD ", "dim"),
        (f"{cpu.load1:.2f}", f"bold {load_color(cpu.load1, len(cpu.per_core))}"),
        "  ",
        (f"{cpu.load5:.2f}", f"bold {load_color(cpu.load5, len(cpu.per_core))}"),
        "  ",
        (f"{cpu.load15:.2f}", f"bold {load_color(cpu.load15, len(cpu.per_core))}")
    )

    title = Text("THERMINAL", style="bold cyan")
    time_str = datetime.now().strftime("%H:%M:%S")

    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=1)
    table.add_column(justify="center", ratio=1)
    table.add_column(justify="right", ratio=1)

    table.add_row(
        Text.assemble((hostname, "bold white"), ("  •  ", "dim"), ("UP ", "bold dim"), (up_str, "bold cyan")),
        title,
        Text.assemble((time_str, "dim"), ("  •  ", "dim"), load_text)
    )

    return Panel(table, box=box.HEAVY, border_style="cyan", padding=(0, 1))


def render_alerts(issues: list[str]) -> Optional[Panel]:
    """Only shown when there are problems."""
    if not issues:
        return None

    text = Text("⚠ ", style="bold yellow")
    text.append("  ".join(issues), style="yellow")
    return Panel(text, box=box.ROUNDED, border_style="yellow", padding=(0, 1))


def render_cpu_panel(cpu: CpuSnapshot) -> Panel:
    """CPU panel with favored core intelligence."""
    cores = len(cpu.per_core)
    physical = psutil.cpu_count(logical=False) or cores // 2
    model = get_cpu_model()
    freq = cpu.freq_mhz / 1000

    # Favored core percentages
    hot_freq: dict[int, float] = {}
    if _hottest_history:
        ctr = Counter(_hottest_history)
        total = len(_hottest_history)
        hot_freq = {k: v / total for k, v in ctr.items()}

    # Core utilization grid + temps
    core_table = Table.grid(padding=(0, 1))
    core_table.add_column(justify="right", width=7)
    core_table.add_column(width=9)
    core_table.add_column(justify="right", width=7)
    core_table.add_column(width=9)

    for i in range(0, cores, 2):
        phys = i // 2
        freq_pct = hot_freq.get(phys, 0.0)

        lbl_style = "bold red" if freq_pct > 0.55 else ("bold orange1" if freq_pct > 0.28 else "dim")
        lbl_style2 = lbl_style

        temp_str = ""
        temp_style = "dim"
        if cpu.temp_cores and phys < len(cpu.temp_cores):
            _, t = cpu.temp_cores[phys]
            c = temp_color(t)
            temp_str = f"{int(round(t))}°"
            temp_style = f"bold {c}"

        label0 = Text.assemble((f"C{i} ", lbl_style), (temp_str, temp_style))
        label1 = Text.assemble((f"C{i+1} ", lbl_style2), (temp_str, temp_style))

        core_table.add_row(
            label0, mini_bar(cpu.per_core[i]),
            label1, mini_bar(cpu.per_core[i+1])
        )

    # Package + summary
    pkg_line = Text("")
    if cpu.temp_package is not None:
        c = temp_color(cpu.temp_package)
        pkg_line = Text.assemble(
            ("PKG ", "dim"),
            (f"{cpu.temp_package:5.1f}°C", f"bold {c}")
        )

    # Favored summary
    favored = Text("")
    if hot_freq:
        leader = max(hot_freq, key=hot_freq.get)
        pct = hot_freq[leader] * 100
        col = "red" if pct > 50 else ("yellow" if pct > 28 else "cyan")
        favored = Text.assemble(
            ("Favored: ", "dim"),
            (f"C{leader} ({pct:.0f}%)", f"bold {col}")
        )

    first = Text.assemble(
        (model, "bold white"), ("  ", "dim"),
        (f"{physical}c/{cores}t", "cyan"), ("  ", "dim"),
        (f"{freq:.2f} GHz", "cyan")
    )

    content = Group(
        first,
        Text(""),
        make_bar(cpu.overall, width=16, label="TOTAL"),
        Text(""),
        pkg_line,
        favored,
        Text(""),
        Text("CORES (temps per physical core)", style="bold dim"),
        core_table,
    )

    return Panel(content, title="[bold cyan]CPU[/]", border_style="blue", box=box.ROUNDED, padding=(0, 1))


def render_gpu_panel(gpus: list[GpuInfo]) -> Panel:
    if not gpus:
        return Panel(Text("No NVIDIA GPU detected", style="dim"), title="[bold]GPU[/]", border_style="dim", box=box.ROUNDED)

    sections = []
    for g in gpus:
        # Beautiful GPU name line with CUDA cores + SM count when available
        name = Text(g.name, style="bold white")
        if g.cuda_cores > 0 and g.sm_count > 0:
            name.append("  •  ", style="dim")
            name.append(f"{g.cuda_cores}c", style="cyan")
            name.append(f" ({g.sm_count} SMs)", style="dim")
        elif g.cuda_cores > 0:
            name.append("  •  ", style="dim")
            name.append(f"{g.cuda_cores}c", style="cyan")
        elif g.sm_count > 0:
            name.append("  •  ", style="dim")
            name.append(f"{g.sm_count} SMs", style="cyan")

        util_bar = make_bar(g.util, width=15, label="UTIL")
        mem_pct = (g.mem_used / g.mem_total * 100) if g.mem_total > 0 else 0
        mem_bar = make_bar(mem_pct, width=15, label="VRAM")

        temp = Text.assemble(("TEMP ", "dim"), (f"{g.temp:5.1f}°C", f"bold {temp_color(g.temp, True)}"))
        pwr = Text.assemble(("PWR  ", "dim"), (f"{g.power:5.1f}W", f"bold {power_color(g.power, g.power_limit)}"))
        if g.power_limit > 0:
            pwr.append(f" / {int(g.power_limit)}W", style="dim")

        fan = Text(f"FAN  {g.fan:5.1f}%", style="cyan") if g.fan >= 0 else Text("")

        clock = Text("")
        if g.sm_clock_mhz:
            clk = f"{g.sm_clock_mhz} MHz"
            if g.throttle_reasons and g.throttle_reasons != "None":
                clk += f"  [throttled: {g.throttle_reasons}]"
            clock = Text.assemble(("CLOCK ", "dim"), (clk, "yellow" if "throttled" in g.throttle_reasons.lower() else "cyan"))

        sections.append(Group(
            name, Text(""),
            util_bar, mem_bar,
            Text(f"{g.mem_used/1024:.1f} / {g.mem_total/1024:.1f} GB", style="dim"),
            Text(""), temp, pwr, clock, fan
        ))
        if len(gpus) > 1:
            sections.append(Text(""))

    return Panel(Group(*sections), title="[bold green]GPU[/]", border_style="green", box=box.ROUNDED, padding=(0, 1))


def render_system_panel(stats: SystemStats) -> Panel:
    ram = make_bar(stats.ram_pct, width=15, label="RAM")
    disk = make_bar(stats.disk_pct, width=15, label="DISK")
    io = make_bar(stats.iowait, width=15, label="IOWAIT")

    swap_lines = []
    if stats.swap_total > 0.5:
        swap = make_bar(stats.swap_pct, width=15, label="SWAP")
        swap_lines = [swap, Text(f"{stats.swap_used:.1f} / {stats.swap_total:.1f} GB", style="dim"), Text("")]

    fds_str = f"{stats.open_fds:,}" + (f" / {stats.fds_max:,}" if isinstance(stats.fds_max, int) else " open fds")

    content = Group(
        ram, Text(f"{stats.ram_used:.1f} / {stats.ram_total:.1f} GB", style="dim"),
        *swap_lines,
        disk, Text(f"{stats.disk_used:.1f} / {stats.disk_total:.1f} GB", style="dim"),
        io, Text(f"{stats.iowait:.1f}% CPU waiting on I/O", style="dim"),
        Text(""),
        Text.assemble(("PROCS ", "dim"), (f"A:{stats.process_total}|{stats.process_running}  ", "cyan"),
                      (f"S:{stats.system_total}|{stats.system_running}  ", "cyan"),
                      (f"U:{stats.user_total}|{stats.user_running}", "cyan")),
        Text(f"FILES  {fds_str}", style="dim"),
    )

    return Panel(content, title="[bold magenta]SYSTEM[/]", border_style="magenta", box=box.ROUNDED, padding=(0, 1))


def render_processes_panel(cpu_procs: list[CpuProcess], gpu_procs: list[GpuProcess]) -> Panel:
    lines: list[RenderableType] = []

    if cpu_procs:
        lines.append(Text("Top CPU", style="bold yellow"))
        t = Table.grid(padding=(0, 0))
        t.add_column(width=16)
        t.add_column(width=7, style="yellow")
        t.add_column(width=6, justify="right", style="cyan")
        t.add_column(width=8)
        for p in cpu_procs[:5]:
            mem = f"{p.mem_mib/1024:.1f}G" if p.mem_mib > 1024 else f"{p.mem_mib:.0f}M"
            t.add_row(p.name, f"{p.cpu:5.1f}%", mem, mini_bar(p.cpu, 8))
        lines.append(t)

    if gpu_procs:
        if lines: lines.append(Text(""))
        lines.append(Text("Top GPU", style="bold green"))
        t = Table.grid(padding=(0, 0))
        t.add_column(width=15)                    # name
        t.add_column(width=6, justify="right", style="cyan")   # memory
        t.add_column(width=7, style="green")      # sm%
        t.add_column(width=8)                     # bar
        for p in gpu_procs[:5]:
            mem_str = f"{p.mem_mib/1024:.1f}G" if p.mem_mib >= 1024 else (f"{p.mem_mib:.0f}M" if p.mem_mib > 1 else "")
            sm_str = f"sm{p.sm_util:4.0f}%" if p.sm_util > 0.5 else ""
            bar = mini_bar(max(p.sm_util, (p.mem_mib / 2048.0 * 100) if p.mem_mib > 0 else 0), 8)
            t.add_row(p.name, mem_str, sm_str, bar)
        lines.append(t)

    if not lines:
        lines = [Text("No significant activity", style="dim")]

    return Panel(Group(*lines), title="[bold green]PROCESSES[/]", border_style="green", box=box.ROUNDED, padding=(0, 1))


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def build_layout(cpu: CpuSnapshot, gpus: list[GpuInfo], gpu_procs: list[GpuProcess],
                 cpu_procs: list[CpuProcess], sysstats: SystemStats,
                 issues: list[str], interval: float) -> Layout:
    """Build the full screen layout.

    We deliberately avoid ever creating a Layout(name='alerts') with size=0.
    This prevents Rich from leaking its internal debug representation.
    """
    layout = Layout()

    # Build the vertical stack dynamically
    parts = [
        Layout(name="header", size=3),
    ]

    alerts_panel = render_alerts(issues)
    if alerts_panel:
        alerts_layout = Layout(name="alerts", size=3)
        alerts_layout.update(alerts_panel)
        parts.append(alerts_layout)

    parts.append(Layout(name="body"))
    parts.append(Layout(name="footer", size=1))

    layout.split_column(*parts)

    # Header
    layout["header"].update(render_header(cpu, shutil.get_terminal_size((100, 40)).columns))

    # Body (always present)
    left = Layout(name="left")
    right = Layout(name="right")
    layout["body"].split_row(left, right)

    left.split_column(
        Layout(name="cpu", ratio=3),
        Layout(name="system", ratio=2),
    )
    right.split_column(
        Layout(name="gpu", ratio=3),
        Layout(name="procs", ratio=2),
    )

    left["cpu"].update(render_cpu_panel(cpu))
    left["system"].update(render_system_panel(sysstats))
    right["gpu"].update(render_gpu_panel(gpus))
    right["procs"].update(render_processes_panel(cpu_procs, gpu_procs))

    # Footer
    gpu_status = "NVIDIA" if gpus else "CPU-only"
    footer = Text.assemble(
        ("q", "bold yellow"), " quit   ",
        ("+/-", "bold yellow"), " rate   ",
        (f"{interval:.1f}s", "cyan"), "   •   ",
        (gpu_status, "dim")
    )
    layout["footer"].update(footer)

    return layout


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

    def get_key(self) -> Optional[str]:
        if not sys.stdin.isatty():
            return None
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if dr else None


def run(interval: float = DEFAULT_INTERVAL, debug: bool = False):
    key_reader = KeyReader()

    with Live(
        build_layout(
            get_cpu_snapshot(), parse_nvidia_smi(debug=debug), [], [],
            get_system_stats(debug=debug), [], interval
        ),
        console=console,
        screen=True,
        refresh_per_second=8,
        transient=False,
    ) as live:
        with key_reader:
            last = time.monotonic()
            while True:
                key = key_reader.get_key()
                if key in ("q", "Q", "\x03"):
                    break
                if key in ("+", "="):
                    interval = max(MIN_INTERVAL, interval - 0.2)
                if key in ("-", "_"):
                    interval = min(MAX_INTERVAL, interval + 0.2)

                now = time.monotonic()
                if now - last >= interval:
                    cpu = get_cpu_snapshot()
                    gpus = parse_nvidia_smi(debug=debug)
                    gpu_procs = get_gpu_processes(gpus, debug=debug)
                    cpu_procs = get_top_cpu_processes(5, debug=debug)
                    sysstats = get_system_stats(debug=debug)
                    issues = detect_issues(cpu, gpus, sysstats)

                    live.update(build_layout(cpu, gpus, gpu_procs, cpu_procs, sysstats, issues, interval))
                    last = now

                time.sleep(0.04)


def main():
    parser = argparse.ArgumentParser(
        description="Therminal — Professional System Monitor",
        epilog="Example: therminal.py -i 0.5 --debug"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Refresh interval in seconds (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print collection errors to stderr instead of silently ignoring them"
    )
    args = parser.parse_args()

    try:
        run(max(MIN_INTERVAL, min(args.interval, MAX_INTERVAL)), debug=args.debug)
    except KeyboardInterrupt:
        pass
    finally:
        console.show_cursor()
        console.print("\n[dim]Therminal exited.[/]")


if __name__ == "__main__":
    main()
