# Changelog

## 0.2.0 - 2026-08-20

### CPU policy support

- Discover CPU controls through Linux cpufreq policies, including Intel, AMD, ARM, and other drivers that expose writable policy controls.
- Keep each policy's governor, frequency range, target, pulse duty, state snapshot, restore path, status, and benchmark result separate.
- Apply boost, CPB, EPP, EPB, PM QoS, and cpuidle controls only when the kernel exposes a writable path.

### Target behavior

- Add `--target auto` to start and benchmark commands. Omitting `--target` now uses each policy's active upper frequency limit.
- Keep numeric targets as explicit MHz requests and clamp them within each policy's usable range.
- Return policy targets, clamp reasons, applied controls, and skipped controls through status and IPC reconfiguration responses.
- Remove the fixed 4 GHz target from the bundled systemd unit.

### Safety and verification

- Preflight every planned write before the first mutation and roll back completed actions when a later action fails.
- Keep PM QoS device access and cpuidle fallback inside the same startup transaction as cpufreq changes.
- Add policy fixtures and contracts for shared policies, mixed limits, missing controls, write failures, rollback, and automatic targets.
- Run 415 automated tests with 97.07 percent coverage. Physical hardware smoke tests remain necessary before calling an untested driver verified.

## 0.1.0

- First public release.
- Adds CPU boost control, thermal limits, state restoration, a systemd unit, and a benchmark command.
