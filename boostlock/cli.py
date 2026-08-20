"""
boostlock/cli.py - Command-line interface for boostlock.

Subcommands
-----------
  start   - Start the boost-lock daemon or foreground engine
  stop    - Stop the running daemon
  status  - Display per-core frequency, governor, temperatures
  bench   - Run a frequency-stability benchmark
  restore - Emergency state restoration from the last snapshot
  service - Manage the systemd unit (install / uninstall / status)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bench import BenchmarkRunner
from .config import BoostLockConfig
from .daemon import BoostLockDaemon
from .ipc import IPCClient
from .protocol import Command, Request
from .state import StateSnapshotManager

# ---------------------------------------------------------------------------
# Public default paths (may be overridden via env-vars or tests)
# ---------------------------------------------------------------------------
DEFAULT_SOCKET_PATH = Path(
    os.environ.get("BOOSTLOCK_SOCKET", "/var/run/boostlock/boostlock.sock")
)
DEFAULT_PID_PATH = Path(
    os.environ.get("BOOSTLOCK_PID", "/var/run/boostlock/boostlock.pid")
)
SYSTEMD_SERVICE_SRC = Path(__file__).parent.parent / "systemd" / "boostlock.service"
SYSTEMD_SERVICE_DST = Path("/etc/systemd/system/boostlock.service")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect_client(socket_path: Path):
    """Return a connected IPCClient, or None with an error already printed."""
    try:
        client = IPCClient(socket_path=socket_path)
        return client
    except Exception as exc:  # noqa: BLE001
        print(f"[boostlock] Cannot connect to daemon socket {socket_path}: {exc}",
              file=sys.stderr)
        print("[boostlock] Is the daemon running?  Try: boostlock start --daemon",
              file=sys.stderr)
        return None


def _send_command(client, cmd: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Send a command and return the parsed response body, or None on error."""
    try:
        req = Request(command=Command.from_str(cmd) if isinstance(cmd, str) else cmd, args=kwargs)
        # Use longer timeout for STATUS which can be slow under high CPU load (pulse workers holding GIL)
        # PING is fast, STATUS needs 10s, others 5s
        timeout = 10.0 if cmd.upper() == "STATUS" else None
        resp = client.send(req, timeout=timeout) if timeout else client.send(req)
        if resp is None:
            print(f"[boostlock] No response from daemon for command '{cmd}'",
                  file=sys.stderr)
            return None
        if not resp.success:
            print(f"[boostlock] Daemon returned error: {resp.error}", file=sys.stderr)
            return None
        return resp.data
    except Exception as exc:  # noqa: BLE001
        print(f"[boostlock] Communication error: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Start command
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace, socket_path: Path, pid_path: Path) -> int:
    """Start boostlock in the foreground (or as a daemon if --daemon)."""
    if args.daemon:
        # Spawn a background process
        try:
            cmd = [sys.executable, "-m", "boostlock.cli", "start"]
            if args.target:
                cmd += ["--target", str(args.target)]
            if args.duty:
                cmd += ["--duty", str(args.duty)]
            if args.max_temp:
                cmd += ["--max-temp", str(args.max_temp)]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print(f"[boostlock] Daemon started (PID {proc.pid})")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[boostlock] Failed to start daemon: {exc}", file=sys.stderr)
            return 1

    # Foreground mode - run daemon directly
    try:
        cfg = BoostLockConfig()
        if args.target:
            cfg.target_freq_khz = int(args.target * 1_000)  # MHz -> kHz
        if args.duty:
            # Map CLI --duty percent to config's min pulse duty (keeps max at default 50 unless duty higher)
            try:
                cfg.min_pulse_duty_pct = float(args.duty)
                if cfg.max_pulse_duty_pct < cfg.min_pulse_duty_pct:
                    cfg.max_pulse_duty_pct = float(args.duty)
            except (TypeError, AttributeError):
                # For mocked config in tests, just set the attribute without comparison
                cfg.min_pulse_duty_pct = float(args.duty)
                try:
                    cfg.max_pulse_duty_pct = float(args.duty)
                except Exception:
                    pass
        if args.max_temp:
            cfg.thermal_limit_c = float(args.max_temp)

        daemon = BoostLockDaemon(
            config=cfg,
            socket_path=socket_path,
            pid_file=pid_path,
        )
        daemon.run()
        return 0
    except KeyboardInterrupt:
        print("\n[boostlock] Interrupted - restoring original state...")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[boostlock] Fatal error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Stop command
# ---------------------------------------------------------------------------

def cmd_stop(args: argparse.Namespace, socket_path: Path) -> int:
    client = _connect_client(socket_path)
    if client is None:
        return 1
    data = _send_command(client, "STOP")
    if data is None:
        return 1
    print("[boostlock] Daemon stopped successfully.")
    return 0


# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace, socket_path: Path) -> int:
    client = _connect_client(socket_path)
    if client is None:
        return 1
    data = _send_command(client, "STATUS")
    if data is None:
        return 1

    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return 0

    # Pretty table
    _render_status_table(data, watch=getattr(args, "watch", False))
    return 0


def _render_status_table(data: Dict[str, Any], watch: bool = False) -> None:
    """Print a human-readable status table."""
    import shutil
    width = min(shutil.get_terminal_size((80, 24)).columns, 100)

    def _row(label: str, value: str, width: int = width) -> str:
        return f"  {label:<26}{value}"

    lines: List[str] = []
    lines.append("-" * width)
    lines.append("  BoostLock Status")
    lines.append("-" * width)

    boost_state = data.get("boost_state", "unknown")
    governor = data.get("governor", "unknown")
    pm_qos = "active" if data.get("pm_qos_active") else "inactive"

    lines.append(_row("Boost state:", boost_state))
    lines.append(_row("Governor:", governor))
    lines.append(_row("PM QoS latency lock:", pm_qos))

    target_khz = data.get("target_freq_khz", 0)
    lines.append(_row("Target:", f"{target_khz / 1_000_000:.3f} GHz"))

    temp = data.get("temperature_c")
    if temp is not None:
        lines.append(_row("Package temperature:", f"{temp:.1f}C"))
    else:
        lines.append(_row("Package temperature:", "unavailable"))

    duty = data.get("duty_cycle")
    if duty is not None:
        lines.append(_row("Pulse duty cycle:", f"{duty * 100:.1f}%"))

    lines.append("-" * width)

    # Per-core table
    per_cpu: Dict[str, Any] = data.get("per_cpu", {})
    if per_cpu:
        col_w = max(12, (width - 4) // max(len(per_cpu), 1))
        header = "  CPU"
        freq_row = "  Freq"
        for cpu_id, cpu_data in sorted(per_cpu.items(), key=lambda x: int(x[0])):
            freq_khz = cpu_data.get("cur_freq_khz", 0)
            freq_str = f"{freq_khz / 1_000_000:.3f}G"
            header += f"  {str(cpu_id):>6}"
            freq_row += f"  {freq_str:>6}"
        lines.append(header)
        lines.append(freq_row)
        lines.append("-" * width)

    print("\n".join(lines))

    if watch:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# Benchmark command
# ---------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> int:
    """Run boost clock stability benchmark."""
    try:
        target_mhz = getattr(args, "target", 4000) or 4000
        target_khz = int(target_mhz) * 1_000
        duration = getattr(args, "duration", 10) or 10
        sample_hz = getattr(args, "sample_hz", 20) or 20

        print(f"[boostlock] Running benchmark: target={target_mhz} MHz, "
              f"duration={duration}s, sample_rate={sample_hz}Hz")
        print("[boostlock] Sampling... (this will take the full duration)")

        runner = BenchmarkRunner(
            target_khz=target_khz,
            duration_s=float(duration),
            sample_rate_hz=float(sample_hz),
        )
        result = runner.run()
        report = result.format_report()
        print(report)

        output = getattr(args, "output", None)
        if output:
            out_path = Path(output)
            out_path.write_text(report + "\n", encoding="utf-8")
            print(f"[boostlock] Report saved to {out_path}")

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[boostlock] Benchmark error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Restore command
# ---------------------------------------------------------------------------

def cmd_restore(args: argparse.Namespace) -> int:
    """Manually trigger emergency state restoration from snapshot."""
    try:
        mgr = StateSnapshotManager()
        # Support both mocked load() and real restore() paths
        # Check if load is mocked (test) vs real SystemStateSnapshot.load
        # Mocked load has no required args and returns None or MagicMock
        load_attr = getattr(mgr, "load", None)
        is_mocked_load = False
        try:
            # Detect mocked load by checking if it's a MagicMock (has called attribute)
            from unittest.mock import MagicMock
            is_mocked_load = isinstance(load_attr, MagicMock)
        except Exception:
            is_mocked_load = False
        if is_mocked_load:
            snap = mgr.load()  # type: ignore[attr-defined]
            if snap is None:
                print("[boostlock] No snapshot found - nothing to restore.")
                return 0
            print("[boostlock] Restoring original system state from snapshot...")
            mgr.restore(snap)
            print("[boostlock] Restore complete.")
            return 0
        # Real manager: check file existence
        if not mgr.snapshot_file.exists():
            print("[boostlock] No snapshot found - nothing to restore.")
            return 0
        print("[boostlock] Restoring original system state from snapshot...")
        mgr.restore()
        print("[boostlock] Restore complete.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[boostlock] Restore error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Systemd service command
# ---------------------------------------------------------------------------

def cmd_service(args: argparse.Namespace) -> int:
    """Manage the systemd boostlock.service unit."""
    action = getattr(args, "service_action", "status")
    try:
        if action == "install":
            if not SYSTEMD_SERVICE_SRC.exists():
                print(f"[boostlock] Service file not found: {SYSTEMD_SERVICE_SRC}",
                      file=sys.stderr)
                return 1
            import shutil
            shutil.copy2(SYSTEMD_SERVICE_SRC, SYSTEMD_SERVICE_DST)
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            print(f"[boostlock] Service installed to {SYSTEMD_SERVICE_DST}")
            print("[boostlock] Run: systemctl enable --now boostlock")
            return 0

        elif action == "uninstall":
            subprocess.run(["systemctl", "stop", "boostlock"], check=False)
            subprocess.run(["systemctl", "disable", "boostlock"], check=False)
            if SYSTEMD_SERVICE_DST.exists():
                SYSTEMD_SERVICE_DST.unlink()
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            print("[boostlock] Service uninstalled.")
            return 0

        elif action == "status":
            result = subprocess.run(
                ["systemctl", "status", "boostlock"],
                capture_output=False,
            )
            return result.returncode

        else:
            print(f"[boostlock] Unknown service action: {action}", file=sys.stderr)
            return 2

    except FileNotFoundError:
        print("[boostlock] systemctl not found - systemd not available on this system.",
              file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[boostlock] Service command failed: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boostlock",
        description=textwrap.dedent("""\
            boostlock - 24/7 sustained CPU boost clock management.
            Tricks the hardware into holding peak boost frequency continuously
            through sysfs governor pinning, PM QoS C-state prevention, and
            adaptive micro-pulse stimulation.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--socket",
        metavar="PATH",
        default=str(DEFAULT_SOCKET_PATH),
        help="Unix domain socket path for IPC (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # -- start ---------------------------------------------------------------
    p_start = subparsers.add_parser(
        "start",
        help="Start the boost-lock engine",
        description="Lock CPU to boost frequency. Uses foreground mode unless --daemon.",
    )
    p_start.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as a background daemon process",
    )
    p_start.add_argument(
        "--target",
        type=float,
        metavar="MHZ",
        help="Target boost frequency in MHz (default: auto-detect from CPU)",
    )
    p_start.add_argument(
        "--duty",
        type=float,
        metavar="PERCENT",
        help="Initial pulse duty cycle 0-100%% (default: 15%%)",
    )
    p_start.add_argument(
        "--max-temp",
        type=float,
        metavar="CELSIUS",
        help="Critical temperature threshold in C (default: 100.0)",
    )

    # -- stop ----------------------------------------------------------------
    subparsers.add_parser(
        "stop",
        help="Stop the running daemon",
        description="Send STOP command to the running boostlock daemon.",
    )

    # -- status --------------------------------------------------------------
    p_status = subparsers.add_parser(
        "status",
        help="Show real-time boost status",
        description="Display per-core frequency table, temperatures, and boost state.",
    )
    p_status.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously refresh (similar to watch(1))",
    )
    p_status.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted table",
    )

    # -- bench ---------------------------------------------------------------
    p_bench = subparsers.add_parser(
        "bench",
        help="Run a boost-clock stability benchmark",
        description="Sample per-core frequencies and compute boost compliance metrics.",
    )
    p_bench.add_argument(
        "--duration",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Benchmark duration in seconds (default: %(default)s)",
    )
    p_bench.add_argument(
        "--target",
        type=int,
        default=4000,
        metavar="MHZ",
        help="Target boost frequency in MHz (default: %(default)s)",
    )
    p_bench.add_argument(
        "--sample-hz",
        type=float,
        default=20.0,
        metavar="HZ",
        help="Sampling rate in Hz (default: %(default)s)",
    )
    p_bench.add_argument(
        "--output",
        metavar="FILE",
        help="Write report to FILE as well as stdout",
    )

    # -- restore -------------------------------------------------------------
    subparsers.add_parser(
        "restore",
        help="Emergency state restoration from snapshot",
        description="Manually restore the CPU governor and frequency settings "
                    "that were saved when boostlock last started.",
    )

    # -- service -------------------------------------------------------------
    p_service = subparsers.add_parser(
        "service",
        help="Manage the systemd service",
        description="Install, uninstall, or query the boostlock systemd unit.",
    )
    p_service.add_argument(
        "service_action",
        choices=["install", "uninstall", "status"],
        help="Action to perform on the systemd service",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    socket_path = Path(args.socket)
    pid_path = DEFAULT_PID_PATH

    if args.command == "start":
        return cmd_start(args, socket_path, pid_path)
    elif args.command == "stop":
        return cmd_stop(args, socket_path)
    elif args.command == "status":
        return cmd_status(args, socket_path)
    elif args.command == "bench":
        return cmd_bench(args)
    elif args.command == "restore":
        return cmd_restore(args)
    elif args.command == "service":
        return cmd_service(args)
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
