# Changelog

## 0.2.0 - 2026-08-20

- Per-policy cpufreq handling. Discovers `policy*` dirs, keeps governor/limits/target/duty/restore separate for each. Falls back to `cpuN/cpufreq` when policies are not there. Drivers that expose writable policy controls work, that includes intel_pstate, amd-pstate, acpi-cpufreq and generic.
- `auto` target. `--target auto` (the default now if you omit `--target`) uses each policy's current max. Numeric targets are clamped per policy. Status and bench now show requested vs effective target plus what got applied or skipped.
- Safer startup. Preflight all writes before touching anything, roll back in reverse order if something fails mid-apply. PM QoS and the cpuidle fallback are part of the same transaction.
- Systemd unit no longer ships a fixed 4 GHz target.
- More tests, fixtures for shared policies, mixed limits, missing controls and rollback. Still need real hardware checks for new drivers.

## 0.1.0

- First release. Governor and boost control, thermal limits, state restore, systemd unit and `bench`.
