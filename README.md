# BoostLock

Keeps CPU boost clocks from dropping when the system is idle. Sets the governor and frequency limits, holds a PM QoS latency lock if the kernel exposes one, and runs a tiny pulse workload to keep the cores awake.

Needs root. It will use more power and run hotter at idle, so dont use it on a machine with bad cooling.

## Requirements

- Linux 5.4+ with cpufreq at `/sys`
- Python 3.10+
- Root for the cpufreq and `/dev/cpu_dma_latency` writes

I test against whatever drivers expose writable cpufreq policies. `intel_pstate`, `amd-pstate`, `acpi-cpufreq` and the generic policy driver all work the same way here, if the governor and limits are writable it is supported. Boost, CPB, EPP, EPB, PM QoS and cpuidle are optional. If the kernel does not expose them BoostLock just skips them. CPU model names are only used for display.

Tested in fixtures for Intel, AMD and ARM. Policy support does not mean the hardware will actually hit a given frequency, that is up to the CPU and firmware.

## How it handles policies

On start it looks for `/sys/devices/system/cpu/cpufreq/policy*` and groups the CPUs. If that directory is not there it falls back to `cpuN/cpufreq` and dedupes the aliases.

Each policy gets its own limits and target. `auto` means use that policy's current max, a number like `3900` gets clamped to what that policy allows. On a heterogenous box the P and E policies can end up with different targets, which is intentional.

It preflights every write first. If anything cannot be opened for writing it bails without changing anything. If a write fails partway through it rolls back what it already did.

## Install

```bash
python3 -m pip install .
boostlock --help
```

## Quick start

Try it in the foreground first:

```bash
sudo boostlock start --target auto --duty 5 --max-temp 90
```

If that looks good, run it as a daemon:

```bash
sudo boostlock start --daemon --target auto --duty 5 --max-temp 90
sudo boostlock status
```

Stop and restore:

```bash
sudo boostlock stop
sudo boostlock restore
```

`--target` is MHz or `auto`. If you leave it off you get `auto`. Numbers are per-policy and get clamped, so `--target 3900` means "try 3900 on each policy if that policy can do it".

## What it actually does on start

Snapshots the values it is about to change, checks all the paths are writable, then applies the governor, frequency limits and any boost/CPB/EPP/EPB bits it can. PM QoS tries `/dev/cpu_dma_latency` first, then the cpuidle disable fallback if every fallback path is writable. If neither works it just runs without it.

Thermal checks, pulse threads and the status socket only start if that whole transaction went through. On exit it restores the governor, limits, boost bits and cpuidle state. Normal signals are handled. `kill -9` cannot clean up so run `sudo boostlock restore` after that.

## Status

```bash
sudo boostlock status --json
```

That dumps each policy with its cpus, driver, requested vs effective target, why it clamped, and which controls were applied or skipped. Handy when you have P and E cores on different drivers.

## Thermals

Defaults: starts backing off at 90C, pauses at 100C, resumes below 85C. Use a lower `--max-temp` if you want it to run cooler.

```bash
sudo boostlock status
sudo boostlock bench --duration 10 --target auto
```

Bench just samples `scaling_cur_freq` and checks it against the effective target for each policy. It shows what the kernel reported during that window, not a guarantee.

## Systemd

```bash
sudo boostlock service install
sudo systemctl enable --now boostlock
sudo systemctl status boostlock
```

The bundled unit runs `/usr/local/bin/boostlock` with no fixed target so it uses `auto`. Make sure that path is right for your install before you enable it.

## Troubleshooting

Run with `sudo`. Most of these files reject unprivileged writes.

`status` cannot connect usually means the daemon is not running, or `BOOSTLOCK_SOCKET` differs between the two commands.

If the machine lost power, run `sudo boostlock restore` before starting again.

## Development

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

Tests use temp directories for sysfs and sockets, no root needed.

## License

MIT, see [LICENSE](LICENSE). History is in [CHANGELOG.md](CHANGELOG.md). For security issues use the repo's private vulnerability reporting, see [SECURITY.md](SECURITY.md).
