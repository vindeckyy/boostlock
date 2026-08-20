# BoostLock

BoostLock is a Linux command-line tool for keeping CPU boost available during idle periods. It configures `cpufreq`, holds a PM QoS latency constraint, and runs a small configurable workload to keep selected CPUs active.

It requires root and can raise idle power use and temperature. Use it on a system with adequate cooling.

## Requirements

- Linux 5.4 or later with `cpufreq` sysfs mounted at `/sys`
- Python 3.10 or later
- Root access for the CPU and PM QoS settings

BoostLock bases its decisions on Linux cpufreq policy capabilities. CPU model names remain display metadata. It needs a writable governor and policy frequency limits. Boost, CPB, EPP, EPB, PM QoS, and cpuidle controls are optional and reported as skipped when the kernel does not expose them.

| Linux capability | Result |
| --- | --- |
| Writable cpufreq policy governor and limits | Supported |
| `intel_pstate`, `amd-pstate`, `acpi-cpufreq`, or an architecture-neutral policy driver | Eligible through the same policy checks |
| Boost, CPB, EPP, EPB, PM QoS, or cpuidle | Applied only when available and writable |
| Firmware-only frequency management or no usable policy | Start fails before changing settings |

Intel, AMD, and ARM policy fixtures are covered by the automated tests. A policy-capable driver is eligible, but that does not promise a particular boost frequency.

## CPU policy support

At startup, BoostLock discovers `/sys/devices/system/cpu/cpufreq/policy*` entries and groups their member CPUs. It falls back to `cpuN/cpufreq` only when policy directories are unavailable, then removes duplicate aliases.

Each usable policy gets its own frequency bounds and effective target. In automatic mode, that target is the policy's active upper limit. A numeric request is clamped within that policy's usable range. A mixed-capacity machine can therefore use different effective targets without one policy inheriting another policy's limit.

Before BoostLock changes a setting, it builds the full policy plan, saves the affected values, and opens every planned path for writing. A preflight failure leaves settings unchanged. A later write failure restores completed actions in reverse order and reports any restore failure.

## Install

Install from a checkout:

```bash
python3 -m pip install .
```

Run the help command to confirm the install:

```bash
boostlock --help
```

## Start and stop

Run in the foreground while testing a new machine:

```bash
sudo boostlock start --target auto --duty 5 --max-temp 90
```

Start a background daemon after checking the foreground run:

```bash
sudo boostlock start --daemon --target auto --duty 5 --max-temp 90
sudo boostlock status
```

Stop the daemon and restore the saved CPU state:

```bash
sudo boostlock stop
sudo boostlock restore
```

`--target` accepts a MHz value or `auto`. Omitting it selects `auto`, which uses each policy's active upper limit. Numeric targets remain explicit requests and are clamped separately for every policy. `--duty` is the initial pulse duty percentage.

Use an explicit request only when you need one:

```bash
sudo boostlock start --target 3900 --duty 5 --max-temp 90
```

## What happens when it starts

`BoostLockDaemon.start()` creates a policy plan, snapshots the values it will change, and preflights each planned write before the first mutation. It then applies the configured governor, policy frequency limits, and any available boost controls as one transaction.

PM QoS uses `/dev/cpu_dma_latency` when it can preflight the device. Its cpuidle fallback is selected only when every planned fallback path can be opened. If neither route is usable, BoostLock records the skip and continues without PM QoS. Thermal monitoring, pulse workers, and the Unix socket start only after the transaction succeeds.

The daemon restores the saved governor, frequency limits, boost settings, EPP or EPB settings, and cpuidle state when it stops. Signal handlers cover normal termination. `kill -9` cannot run cleanup, so run `sudo boostlock restore` after a forced kill.

## Reading policy status

Use JSON status output to inspect the resolved plan:

```bash
sudo boostlock status --json
```

Each policy entry includes its identifier, member CPUs, driver, requested target, effective target, clamp reason, applied controls, and skipped controls. This is useful on systems with performance and efficiency policies or a mixture of CPU drivers.

## Thermal behavior

The default thermal guard starts reducing pulse duty at 90 C, pauses it at 100 C, and resumes only after the temperature falls below 85 C. Set a lower `--max-temp` for a cooler system.

Check the daemon before leaving it running:

```bash
sudo boostlock status
sudo boostlock bench --duration 10 --target auto
```

The benchmark samples `scaling_cur_freq` and evaluates each selected policy against its own effective target. It reports what the kernel exposed during that run. It does not establish a stable frequency for every workload.

## Systemd

Install the bundled unit, then enable it:

```bash
sudo boostlock service install
sudo systemctl enable --now boostlock
sudo systemctl status boostlock
```

The unit starts `/usr/local/bin/boostlock` without a fixed target, so it uses automatic per-policy targets. Verify that path exists on the target machine before enabling the service.

## Maintainer

BoostLock is maintained by [vindeckyy](https://github.com/vindeckyy).

## Troubleshooting

Run commands with `sudo`. CPU sysfs files and `/dev/cpu_dma_latency` usually reject unprivileged writes.

If `status` cannot connect, confirm that the daemon is running and that both commands use the same `BOOSTLOCK_SOCKET` value.

If the machine was shut down abruptly, run `sudo boostlock restore` before starting another daemon.

## Development

Run the test suite from a checkout:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

The tests use temporary sysfs and socket paths, so they do not require root.

## License

MIT. See [LICENSE](LICENSE).

Release history is in [CHANGELOG.md](CHANGELOG.md). Security reports belong in the repository's private vulnerability reporting flow. See [SECURITY.md](SECURITY.md).
