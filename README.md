# BoostLock

BoostLock is a Linux command-line tool for keeping CPU boost available during idle periods. It configures `cpufreq`, holds a PM QoS latency constraint, and runs a small configurable workload to keep selected CPUs active.

It requires root and can raise idle power use and temperature. Use it on a system with adequate cooling.

## Requirements

- Linux 5.4 or later with `cpufreq` sysfs mounted at `/sys`
- Python 3.10 or later
- Root access for the CPU and PM QoS settings

BoostLock works best on systems that expose `scaling_governor`, boost controls, and `/dev/cpu_dma_latency`. Missing optional kernel files are skipped, and the requested boost frequency remains subject to the CPU and kernel.

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
sudo boostlock start --target 3900 --duty 5 --max-temp 90
```

Start a background daemon after checking the foreground run:

```bash
sudo boostlock start --daemon --target 3900 --duty 5 --max-temp 90
sudo boostlock status
```

Stop the daemon and restore the saved CPU state:

```bash
sudo boostlock stop
sudo boostlock restore
```

`--target` uses MHz. `--duty` is the initial pulse duty percentage. Start with a low duty value and increase it only if the reported frequency drops below the target.

## What happens when it starts

`BoostLockDaemon.start()` saves the current CPU state before it changes anything. It then sets the configured governor and boost controls, opens the PM QoS latency device, starts thermal monitoring and pulse workers, and opens the Unix socket used by `status` and `stop`.

The daemon restores the saved governor, frequency limits, boost settings, EPP or EPB settings, and cpuidle state when it stops. Signal handlers cover normal termination. `kill -9` cannot run cleanup, so run `sudo boostlock restore` after a forced kill.

## Thermal behavior

The default thermal guard starts reducing pulse duty at 90 C, pauses it at 100 C, and resumes only after the temperature falls below 85 C. Set a lower `--max-temp` for a cooler system.

Check the daemon before leaving it running:

```bash
sudo boostlock status
sudo boostlock bench --duration 10 --target 3900
```

The benchmark samples `scaling_cur_freq`. It reports what the kernel exposed during that run. It does not establish a stable frequency for every workload.

## Systemd

Install the bundled unit, then enable it:

```bash
sudo boostlock service install
sudo systemctl enable --now boostlock
sudo systemctl status boostlock
```

The unit starts `/usr/local/bin/boostlock`. Verify that path exists on the target machine before enabling the service.

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
