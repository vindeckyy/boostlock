# Changelog

## Unreleased

- Add policy-based CPU support for Intel, AMD, ARM, and other Linux cpufreq drivers with writable policy controls.
- Use per-policy targets, preflighted writes, rollback, and optional-control reporting.
- Add `--target auto` and make omitted CLI and service targets automatic per policy.
- Do not claim a fixed boost frequency for capability-eligible drivers that have not been tested on hardware.

## 0.1.0

- First public release.
- Adds CPU boost control, thermal limits, state restoration, a systemd unit, and a benchmark command.
