#!/usr/bin/env python3
"""Therminal — beautiful GPU + CPU thermal + load TUI."""

import argparse
import csv
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
    temp_cores: list[float]
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

    # Temperatures via psutil
    temps = psutil.sensors_temperatures()
    temp_package = None
    temp_cores: list[float] = []

    if "coretemp" in temps:
        for entry in temps["coretemp"]:
            label = (entry.label or "").lower()
            if "package" in label or entry.label == "Package id 0":
                temp_package = entry.current
            elif "core" in label:
                temp_cores.append(entry.current)

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
                "temperature.gpu,power.draw,power.limit,fan.speed",
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
            if len(row) < 9:
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

    # Process count
    try:
        stats["process_count"] = len(psutil.pids())
    except Exception:
        stats["process_count"] = 0

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
    up_str = f"{up.days}d {up.seconds // 3600}h {(up.seconds % 3600) // 60}m"

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
            (up_str, "cyan"),
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


def render_cpu_panel(cpu: CpuSnapshot) -> Panel:
    """CPU panel: load, per-core, temperatures, frequency."""
    cores = len(cpu.per_core)
    physical = psutil.cpu_count(logical=False) or cores // 2

    # Overall big bar
    overall_bar = make_bar(cpu.overall, width=17, color=util_color(cpu.overall), label="TOTAL")

    # Per-core compact grid (2 columns)
    core_table = Table.grid(padding=(0, 1))
    core_table.add_column(justify="right", style="dim", width=3)
    core_table.add_column()
    core_table.add_column(justify="right", style="dim", width=3)
    core_table.add_column()

    for i in range(0, cores, 2):
        row: list[Any] = [f"C{i}", small_bar(cpu.per_core[i])]
        if i + 1 < cores:
            row.extend([f"C{i+1}", small_bar(cpu.per_core[i + 1])])
        else:
            row.extend(["", ""])
        core_table.add_row(*row)

    # Temperatures
    temp_lines: list[RenderableType] = []
    if cpu.temp_package is not None:
        c = temp_color(cpu.temp_package)
        temp_lines.append(
            Text.assemble(
                ("PKG  ", "dim"),
                (f"{cpu.temp_package:5.1f}°C", f"bold {c}"),
            )
        )

    if cpu.temp_cores:
        avg = sum(cpu.temp_cores) / len(cpu.temp_cores)
        c = temp_color(avg)
        temp_lines.append(
            Text.assemble(
                ("AVG  ", "dim"),
                (f"{avg:5.1f}°C", f"bold {c}"),
                (f"   max {max(cpu.temp_cores):.0f}°C", "dim"),
            )
        )

    # Frequency + memory on one line
    freq_text = Text.assemble(
        ("FREQ  ", "dim"),
        (f"{cpu.freq_mhz / 1000:.2f} GHz", "cyan"),
    )

    model = get_cpu_model()
    first_line = Text.assemble(
        (model, "bold white"),
        ("  •  ", "dim"),
        (f"{cores}t • {physical}c", "dim"),
    )

    content = Group(
        first_line,
        Text(""),
        overall_bar,
        Text(""),
        Text("CORES", style="bold dim"),
        core_table,
        Text(""),
        Text("TEMPS", style="bold dim"),
        *temp_lines,
        Text(""),
        freq_text,
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
        Text(""),
        *swap_lines,
        disk_bar,
        Text(f"{s['disk_used']:.1f} / {s['disk_total']:.1f} GB", style="dim"),
        Text(""),
        iowait_bar,
        Text(f"{s['iowait']:.1f}% CPU waiting on I/O", style="dim"),
        Text(""),
        file_bar,
        Text(files_text, style="dim"),
        Text(""),
        Text(f"PROCS: {s['process_count']}", style="dim"),
    )

    return Panel(
        content,
        title="[bold magenta]SYSTEM[/]",
        border_style="magenta",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_processes_panel(cpu_procs: list[CpuProcess], gpu_procs: list[GpuProcess]) -> Panel:
    """Dedicated PROCESSES panel — top CPU and GPU consumers side by side."""
    lines: list[RenderableType] = []

    # CPU processes (top 3)
    if cpu_procs:
        lines.append(Text("TOP CPU", style="bold yellow"))
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
        lines.append(Text(""))

    # GPU processes (top 3)
    if gpu_procs:
        lines.append(Text("TOP GPU", style="bold green"))
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

        # Combine GPU name + CUDA cores/SMs on the first line (compact)
        name_parts = [Text(g.name, style="bold white")]

        if g.cuda_cores > 0 and g.sm_count > 0:
            name_parts.extend([
                Text("  •  ", style="dim"),
                Text(f"{g.cuda_cores}c", style="cyan"),
                Text(" (", style="dim"),
                Text(f"{g.sm_count} SMs", style="cyan"),
                Text(")", style="dim"),
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

        gpu_block = Group(
            name_line,
            Text(""),
            util_bar,
            Text(""),
            mem_bar,
            mem_detail,
            Text(""),
            temp_text,
            power_text,
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

    layout["cpu"].update(render_cpu_panel(cpu))
    layout["gpu"].update(render_gpu_panel(gpus, gpu_procs))
    layout["system"].update(render_system_panel())
    layout["processes"].update(render_processes_panel(cpu_procs, gpu_procs))

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

    with Live(
        build_layout(get_cpu_snapshot(), parse_nvidia_smi(), [], [], interval),
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
                    live.update(build_layout(cpu, gpus, gpu_procs, cpu_procs, interval))
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
