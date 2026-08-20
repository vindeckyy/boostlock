"""Red contracts for policy-owned CPU frequency controls.

These tests intentionally name the D1 public behavior before F1-F4 implement it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

import pytest

from boostlock.bench import BenchResult, BenchmarkRunner
from boostlock.cli import build_parser
from boostlock.config import BoostLockConfig
from boostlock.daemon import BoostLockDaemon, DaemonError, DaemonState
from boostlock.engine import AdaptiveDutyController
from boostlock.hardware import CPUVendor, detect_cpu_info
from boostlock.protocol import Command, Request
from boostlock.state import PolicyStateSnapshot, StateSnapshotManager
from boostlock.sysfs import CapabilityState, PolicyApplyAction, SysfsController, SysfsError
from tests.policy_fixtures import PolicySpec, PolicySysfsFixture


def _field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _two_policy_tree(tmp_path: Path) -> PolicySysfsFixture:
    tree = PolicySysfsFixture(tmp_path)
    tree.add_policy(
        PolicySpec(
            "policy0",
            (0, 2),
            driver="arm-cpufreq",
            hardware_min_khz=600_000,
            hardware_max_khz=2_600_000,
            active_max_khz=2_400_000,
            related_cpus="0 2",
        )
    )
    tree.add_policy(
        PolicySpec(
            "policy4",
            (1, 3),
            driver="amd-pstate-epp",
            hardware_min_khz=1_000_000,
            hardware_max_khz=5_200_000,
            active_max_khz=4_400_000,
            affected_cpus="1 3",
            optional_nodes={"energy_performance_preference": "balance_performance"},
        )
    )
    return tree


class TestPolicyDiscoveryContracts:
    def test_discovers_global_policies_from_related_and_affected_cpu_lists(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)

        policies = SysfsController(tree.root).discover_cpufreq_policies()

        assert [_field(policy, "identifier") for policy in policies] == ["policy0", "policy4"]
        assert _field(policies[0], "cpus") == [0, 2]
        assert _field(policies[1], "cpus") == [1, 3]
        assert _field(policies[0], "driver") == "arm-cpufreq"
        assert policies[0].governors == ["performance", "schedutil", "powersave"]

    def test_deduplicates_cpu_aliases_that_resolve_to_one_policy(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        legacy_policy = tree.add_policy(
            PolicySpec("policy0", (0, 2), related_cpus="0 2"),
            global_policy=False,
        )

        policies = SysfsController(tree.root).discover_cpufreq_policies()

        assert not list(tree.policy_root.glob("policy*"))
        assert len(policies) == 1
        assert _field(policies[0], "identifier") == "policy0"
        assert _field(policies[0], "path") == legacy_policy.resolve()
        assert _field(policies[0], "cpus") == [0, 2]

    def test_global_policies_win_over_conflicting_cpu_aliases(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        global_policy = tree.add_policy(
            PolicySpec("policy0", (0,), active_max_khz=2_600_000),
        )
        alias = tree.cpu_root / "cpu0" / "cpufreq"
        alias.unlink()
        alias.mkdir()
        (alias / "scaling_governor").write_text("powersave\n")
        (alias / "scaling_min_freq").write_text("400000\n")
        (alias / "scaling_max_freq").write_text("900000\n")

        policies = SysfsController(tree.root).discover_cpufreq_policies()

        assert len(policies) == 1
        assert _field(policies[0], "path") == global_policy.resolve()
        assert _field(policies[0], "active_max_khz") == 2_600_000

    def test_hardware_metadata_remains_architecture_neutral(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text("processor : 0\nCPU implementer : ARM Ltd.\nmodel name : Cortex-A78\n")

        info = detect_cpu_info(proc_cpuinfo_path=str(cpuinfo), sysfs_root=str(tree.root))

        assert info.vendor == CPUVendor.UNKNOWN
        assert [_field(policy, "identifier") for policy in info.policies] == ["policy0", "policy4"]
        assert _field(info.policies[0], "driver") == "arm-cpufreq"

    def test_hardware_discovery_ignores_model_and_cpu_flags_for_boost_controls(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        tree.policy_root.joinpath("boost").unlink()
        tree.add_policy(PolicySpec("policy0", (0,), hardware_max_khz=3_000_000))
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(
            "processor : 0\n"
            "vendor_id : AuthenticAMD\n"
            "model name : AMD Ryzen 5 4600H with Radeon Graphics\n"
            "cpu MHz : 2500\n"
            "flags : cpb ida hwp hwp_epp epb\n"
        )

        info = detect_cpu_info(proc_cpuinfo_path=str(cpuinfo), sysfs_root=str(tree.root))

        assert info.max_boost_mhz == 3_000.0
        assert info.has_cpb is False
        assert info.has_boost is False
        assert info.has_epp is False
        assert info.has_epb is False

    def test_rejects_a_tree_without_a_usable_policy_before_any_write(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        tree.add_policy(PolicySpec("policy0", (0,), missing_nodes=("scaling_governor",)))
        controller = SysfsController(tree.root)

        with pytest.raises(SysfsError, match="usable policy"):
            controller.discover_cpufreq_policies(require_usable=True)

    def test_missing_optional_nodes_are_reported_without_disabling_an_usable_policy(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        tree.add_policy(
            PolicySpec(
                "policy0",
                (0,),
                optional_nodes={"energy_performance_preference": "performance", "cpb": "1"},
                missing_nodes=("energy_performance_preference", "cpb"),
            )
        )

        policy = SysfsController(tree.root).discover_cpufreq_policies()[0]

        assert _field(policy, "usable") is True
        assert _field(policy, "skipped_controls") == {
            "energy_perf_bias": "node unavailable",
            "energy_performance_preference": "node unavailable",
            "cpb": "node unavailable",
        }

    def test_optional_capability_states_and_writable_paths_are_inventoried(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        policy_dir = tree.add_policy(
            PolicySpec(
                "policy0",
                (0,),
                optional_nodes={"cpb": "", "energy_perf_bias": "6"},
                missing_nodes=("energy_performance_preference",),
            )
        )
        tree.policy_root.joinpath("boost").unlink()

        policy = SysfsController(tree.root).discover_cpufreq_policies()[0]

        assert _field(policy, "capabilities")["boost"] == CapabilityState.UNAVAILABLE
        assert _field(policy, "capabilities")["cpb"] == CapabilityState.UNUSABLE
        assert _field(policy, "capabilities")["energy_performance_preference"] == CapabilityState.UNAVAILABLE
        assert _field(policy, "capabilities")["energy_perf_bias"] == CapabilityState.AVAILABLE
        assert _field(policy, "skipped_controls") == {
            "boost": "node unavailable",
            "cpb": "node unusable",
            "energy_performance_preference": "node unavailable",
        }
        assert _field(policy, "writable_paths") == {
            "governor": policy_dir / "scaling_governor",
            "active_min_frequency": policy_dir / "scaling_min_freq",
            "active_max_frequency": policy_dir / "scaling_max_freq",
            "cpb": policy_dir / "cpb",
            "energy_perf_bias": policy_dir / "energy_perf_bias",
        }

    def test_malformed_required_nodes_and_empty_governors_are_unusable(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        policy_dir = tree.add_policy(PolicySpec("policy0", (0,)))
        (policy_dir / "scaling_min_freq").write_text("broken\n")
        (policy_dir / "scaling_available_governors").write_text("\n")

        policy = SysfsController(tree.root).discover_cpufreq_policies()[0]

        assert _field(policy, "capabilities")["active_min_frequency"] == CapabilityState.UNUSABLE
        assert _field(policy, "capabilities")["available_governors"] == CapabilityState.UNUSABLE
        assert _field(policy, "usable") is False

    def test_policy_without_members_does_not_probe_cpu_energy_perf_bias(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        tree.add_policy(
            PolicySpec("policy0", (0,), related_cpus="invalid"),
            aliases=False,
        )

        policy = SysfsController(tree.root).discover_cpufreq_policies()[0]

        assert _field(policy, "cpus") == []
        assert _field(policy, "capabilities")["energy_perf_bias"] == CapabilityState.UNAVAILABLE


class TestPolicyTransactionContracts:
    def test_plan_skips_unusable_policies_and_invalid_frequency_intervals(self, tmp_path: Path) -> None:
        tree = PolicySysfsFixture(tmp_path)
        tree.add_policy(PolicySpec("policy0", (0,)))
        tree.add_policy(PolicySpec("policy4", (1,), missing_nodes=("scaling_governor",)))
        controller = SysfsController(tree.root)

        plan = controller.build_policy_apply_plan(target_khz=2_000_000, governor="performance")

        assert {action.policy_id for action in plan.actions} == {"policy0"}
        assert plan.skipped_controls["policy4"]["policy"] == "policy unusable"

        incompatible_tree = PolicySysfsFixture(tmp_path / "incompatible")
        incompatible_tree.add_policy(
            PolicySpec(
                "policy0",
                (0,),
                hardware_min_khz=3_000_000,
                hardware_max_khz=3_500_000,
                active_max_khz=2_500_000,
            )
        )
        incompatible_plan = SysfsController(incompatible_tree.root).build_policy_apply_plan(
            target_khz=3_000_000,
        )
        assert incompatible_plan.actions == []
        assert incompatible_plan.skipped_controls["policy0"]["active_min_frequency"] == "frequency limits unavailable"

    def test_plan_clamps_each_policy_snapshots_values_and_skips_unsupported_governors(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        tree.policy_path("policy0", "scaling_available_governors").write_text("schedutil\n")
        controller = SysfsController(tree.root)

        plan = controller.build_policy_apply_plan(target_khz=4_000_000, governor="performance")

        actions = {
            (action.policy_id, action.control): action
            for action in plan.actions
        }
        assert ("policy0", "governor") not in actions
        assert actions[("policy0", "active_min_frequency")].value == "2400000"
        assert actions[("policy0", "active_min_frequency")].original_value == "800000"
        assert actions[("policy4", "governor")].original_value == "schedutil"
        assert actions[("policy4", "active_min_frequency")].value == "4000000"
        assert plan.skipped_controls["policy0"]["governor"] == "governor unavailable"
        assert plan.skipped_controls["policy0"]["cpb"] == "node unavailable"

    def test_plan_includes_optional_controls_and_deduplicates_shared_boost(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        for policy in ("policy0", "policy4"):
            tree.policy_path(policy, "cpb").write_text("0\n")
            tree.policy_path(policy, "energy_perf_bias").write_text("6\n")
        tree.policy_path("policy0", "energy_performance_preference").write_text("balance_power\n")
        controller = SysfsController(tree.root)

        plan = controller.build_policy_apply_plan(
            target_khz=4_000_000,
            governor="performance",
            boost=True,
            cpb=True,
            energy_performance_preference="performance",
        )

        actions = {(action.policy_id, action.control): action for action in plan.actions}
        boost_actions = [action for action in plan.actions if action.control == "boost"]
        assert len(boost_actions) == 1
        assert boost_actions[0].path == tree.policy_root / "boost"
        assert boost_actions[0].value == "1"
        assert boost_actions[0].original_value == "0"
        assert actions[("policy0", "cpb")].value == "1"
        assert actions[("policy4", "cpb")].original_value == "0"
        assert actions[("policy0", "energy_performance_preference")].value == "performance"
        assert actions[("policy4", "energy_performance_preference")].original_value == "balance_performance"
        assert plan.skipped_controls["policy4"]["boost"] == "shared control already planned"
        assert plan.skipped_controls["policy0"]["energy_perf_bias"] == "no configured value"
        assert plan.skipped_controls["policy4"]["energy_perf_bias"] == "no configured value"

    def test_mixed_policy_targets_never_reuse_the_first_policy_limit(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        config = BoostLockConfig(target_frequency_khz=4_000_000)
        daemon = BoostLockDaemon(config=config, sysfs_root=tree.root, dma_latency_path=tmp_path / "dma")

        targets = daemon.resolve_policy_targets()

        assert targets["policy0"]["effective_target_khz"] == 2_400_000
        assert targets["policy4"]["effective_target_khz"] == 4_000_000

    def test_applying_two_policies_never_writes_one_policy_limit_to_another(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        daemon = BoostLockDaemon(
            config=BoostLockConfig(target_frequency_khz=4_000_000),
            sysfs_root=tree.root,
            dma_latency_path=tmp_path / "dma",
        )

        daemon._apply_sysfs_boost_profile()

        assert tree.policy_path("policy0", "scaling_min_freq").read_text() == "2400000\n"
        assert tree.policy_path("policy4", "scaling_min_freq").read_text() == "4000000\n"

    def test_daemon_applies_configured_optional_controls_per_policy(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        for policy in ("policy0", "policy4"):
            tree.policy_path(policy, "cpb").write_text("0\n")
        tree.policy_path("policy0", "energy_performance_preference").write_text("balance_power\n")
        daemon = BoostLockDaemon(
            config=BoostLockConfig(target_frequency_khz=4_000_000, epp="performance"),
            sysfs_root=tree.root,
            dma_latency_path=tmp_path / "dma",
        )

        daemon._apply_sysfs_boost_profile()

        assert (tree.policy_root / "boost").read_text() == "1\n"
        assert tree.policy_path("policy0", "cpb").read_text() == "1\n"
        assert tree.policy_path("policy4", "cpb").read_text() == "1\n"
        assert tree.policy_path("policy0", "energy_performance_preference").read_text() == "performance\n"
        assert tree.policy_path("policy4", "energy_performance_preference").read_text() == "performance\n"

    def test_preflight_opens_all_cpufreq_pm_qos_and_cpuidle_paths_before_writes(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        for policy in ("policy0", "policy4"):
            tree.policy_path(policy, "cpb").write_text("0\n")
        tree.policy_path("policy0", "energy_performance_preference").write_text("balance_power\n")
        idle_paths = tree.add_cpuidle((0, 1), states=(1,))
        dma_latency = tmp_path / "cpu_dma_latency"
        dma_latency.write_bytes(b"\x00\x00\x00\x00")
        controller = SysfsController(tree.root)
        audit = tree.write_opener()

        plan = controller.build_policy_apply_plan(
            target_khz=4_000_000,
            governor="performance",
            boost=True,
            cpb=True,
            energy_performance_preference="performance",
            pm_qos_device=dma_latency,
            cpuidle_fallback_paths=idle_paths,
        )
        controller.preflight_policy_apply_plan(plan, open_for_write=audit)

        expected = {
            tree.policy_path("policy0", "scaling_governor").resolve(),
            tree.policy_path("policy0", "scaling_min_freq").resolve(),
            tree.policy_path("policy4", "scaling_governor").resolve(),
            tree.policy_path("policy4", "scaling_min_freq").resolve(),
            tree.policy_root / "boost",
            tree.policy_path("policy0", "cpb").resolve(),
            tree.policy_path("policy4", "cpb").resolve(),
            tree.policy_path("policy0", "energy_performance_preference").resolve(),
            tree.policy_path("policy4", "energy_performance_preference").resolve(),
            dma_latency.resolve(),
            *(path.resolve() for path in idle_paths),
        }
        assert set(audit.opened_paths) == expected

    def test_preflight_reports_close_errors_before_any_mutation(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        plan = controller.build_policy_apply_plan(target_khz=4_000_000)

        class BrokenHandle:
            def close(self) -> None:
                raise OSError("close failed")

        with pytest.raises(SysfsError, match="close write check.*close failed"):
            controller.preflight_policy_apply_plan(plan, open_for_write=lambda _: BrokenHandle())


    def test_rejected_write_open_leaves_every_policy_unmodified(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        rejected = tree.policy_path("policy4", "scaling_min_freq")
        tree.reject_write_open(rejected)
        controller = SysfsController(tree.root)
        audit = tree.write_opener()
        plan = controller.build_policy_apply_plan(target_khz=4_000_000, governor="performance")
        original = {
            path: path.read_text()
            for path in (
                tree.policy_path("policy0", "scaling_governor"),
                tree.policy_path("policy0", "scaling_min_freq"),
                tree.policy_path("policy4", "scaling_governor"),
                tree.policy_path("policy4", "scaling_min_freq"),
            )
        }

        with pytest.raises(SysfsError, match="policy4.*scaling_min_freq"):
            controller.execute_policy_apply_plan(plan, open_for_write=audit)

        assert {path: path.read_text() for path in original} == original

    def test_late_failure_rolls_back_completed_actions_in_reverse_order_and_reports_rollback_failure(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        plan = controller.build_policy_apply_plan(target_khz=4_000_000, governor="performance")
        writes: list[tuple[str, str]] = []

        def writer(path: Path, value: str) -> None:
            writes.append((path.name, value))
            if path == tree.policy_path("policy4", "scaling_min_freq") and value != "800000":
                raise OSError("apply failed")
            if path == tree.policy_path("policy0", "scaling_governor") and value == "schedutil":
                raise OSError("rollback failed")

        with pytest.raises(SysfsError, match="apply failed.*rollback failed"):
            controller.execute_policy_apply_plan(plan, writer=writer)

        assert writes[-2:] == [("scaling_min_freq", "800000"), ("scaling_governor", "schedutil")]

    def test_rollback_continues_after_an_earlier_compensation_fails(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        plan = controller.build_policy_apply_plan(target_khz=4_000_000, governor="performance")
        writes: list[tuple[Path, str]] = []

        def writer(path: Path, value: str) -> None:
            writes.append((path, value))
            if path == tree.policy_path("policy4", "scaling_min_freq") and value == "4000000":
                raise OSError("apply failed")
            if path == tree.policy_path("policy4", "scaling_governor") and value == "schedutil":
                raise OSError("first rollback failed")

        with pytest.raises(SysfsError, match="apply failed.*first rollback failed"):
            controller.execute_policy_apply_plan(plan, writer=writer)

        assert (tree.policy_path("policy0", "scaling_min_freq"), "800000") in writes
        assert (tree.policy_path("policy0", "scaling_governor"), "schedutil") in writes

    def test_default_executor_writes_a_preflighted_plan(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        plan = controller.build_policy_apply_plan(target_khz=4_000_000, governor="performance")

        controller.execute_policy_apply_plan(plan)

        assert tree.policy_path("policy0", "scaling_governor").read_text() == "performance\n"
        assert tree.policy_path("policy4", "scaling_min_freq").read_text() == "4000000\n"

    def test_optional_action_failure_rolls_back_prior_optional_controls(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        for policy in ("policy0", "policy4"):
            tree.policy_path(policy, "cpb").write_text("0\n")
        tree.policy_path("policy0", "energy_performance_preference").write_text("balance_power\n")
        controller = SysfsController(tree.root)
        plan = controller.build_policy_apply_plan(
            target_khz=4_000_000,
            boost=True,
            cpb=True,
            energy_performance_preference="performance",
        )
        writes: list[tuple[Path, str]] = []
        failed_path = tree.policy_path("policy4", "energy_performance_preference")

        def writer(path: Path, value: str) -> None:
            writes.append((path, value))
            if path == failed_path and value == "performance":
                raise OSError("optional apply failed")

        with pytest.raises(SysfsError, match="optional apply failed"):
            controller.execute_policy_apply_plan(plan, writer=writer)

        assert (tree.policy_path("policy4", "cpb"), "0") in writes
        assert (tree.policy_path("policy0", "energy_performance_preference"), "balance_power") in writes
        assert (tree.policy_root / "boost", "0") in writes

    def test_plan_rejects_missing_writable_paths_and_unreadable_snapshots(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        policy = controller.discover_cpufreq_policies()[0]

        with pytest.raises(SysfsError, match="no writable missing"):
            controller._planned_policy_action(policy, "missing", "value")
        with patch.object(controller, "_read_path", return_value=None), \
             pytest.raises(SysfsError, match="Cannot snapshot"):
            controller._planned_policy_action(policy, "governor", "performance")
        policy.hardware_min_khz = None
        policy.active_min_khz = None
        assert controller._effective_policy_target(policy, 2_000_000) is None
        policy.hardware_min_khz = 3_000_000
        policy.active_min_khz = 800_000
        policy.hardware_max_khz = 3_500_000
        policy.active_max_khz = 2_500_000
        assert controller._effective_policy_target(policy, 3_000_000) is None


class TestDaemonAndStatePolicyContracts:
    def test_daemon_returns_to_stopped_and_releases_lock_after_pm_qos_preflight_failure(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        config = BoostLockConfig(
            target_frequency_khz=4_000_000,
            pid_file=str(tmp_path / "boostlock.pid"),
            socket_path=str(tmp_path / "boostlock.sock"),
            snapshot_path=str(tmp_path / "snapshot.json"),
        )
        dma_latency = tmp_path / "dma"
        dma_latency.write_bytes(b"\x00\x00\x00\x00")
        daemon = BoostLockDaemon(config=config, sysfs_root=tree.root, dma_latency_path=dma_latency)

        with patch.object(daemon.pm_qos, "open_device_for_preflight", side_effect=PermissionError("write-open rejected")):
            with pytest.raises(DaemonError, match="write-open rejected"):
                daemon.start()

        assert daemon.state == DaemonState.STOPPED
        assert not daemon.pid_manager.is_locked
        assert tree.policy_path("policy0", "scaling_min_freq").read_text() == "800000\n"
        assert tree.policy_path("policy4", "scaling_min_freq").read_text() == "800000\n"

    def test_daemon_fallback_failure_restores_cpuidle_and_starts_no_later_subsystem(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        idle_paths = tree.add_cpuidle((0, 1), states=(1,))
        config = BoostLockConfig(
            target_frequency_khz=4_000_000,
            pid_file=str(tmp_path / "boostlock.pid"),
            socket_path=str(tmp_path / "boostlock.sock"),
            snapshot_path=str(tmp_path / "snapshot.json"),
        )
        daemon = BoostLockDaemon(config=config, sysfs_root=tree.root, dma_latency_path=tmp_path / "missing_dma")
        original_write = daemon.pm_qos.sysfs._write_file

        def fail_second_idle_write(path: str, value: str, optional: bool = False) -> bool:
            if path.endswith("cpu1/cpuidle/state1/disable") and value == "1":
                raise PermissionError("fallback write failed")
            return original_write(path, value, optional)

        with patch.object(daemon.pm_qos.sysfs, "_write_file", side_effect=fail_second_idle_write), \
             patch.object(daemon.thermal_guard, "start") as thermal_start, \
             patch.object(daemon.pulse_engine, "start") as pulse_start, \
             patch.object(daemon.ipc_server, "start") as ipc_start:
            with pytest.raises(DaemonError, match="fallback write failed"):
                daemon.start()

        assert daemon.state == DaemonState.STOPPED
        assert not daemon.pid_manager.is_locked
        assert all(path.read_text() == "0\n" for path in idle_paths)
        thermal_start.assert_not_called()
        pulse_start.assert_not_called()
        ipc_start.assert_not_called()

    def test_daemon_reports_pm_qos_execution_and_close_compensation_failures(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        config = BoostLockConfig(
            target_frequency_khz=4_000_000,
            pid_file=str(tmp_path / "boostlock.pid"),
            socket_path=str(tmp_path / "boostlock.sock"),
            snapshot_path=str(tmp_path / "snapshot.json"),
        )
        dma_latency = tmp_path / "dma"
        dma_latency.write_bytes(b"\x00\x00\x00\x00")
        daemon = BoostLockDaemon(config=config, sysfs_root=tree.root, dma_latency_path=dma_latency)

        with patch.object(daemon.pm_qos, "lock", side_effect=OSError("pm qos execution failed")), \
             patch.object(daemon.pm_qos, "release_strict", side_effect=OSError("pm qos close failed")), \
             patch.object(daemon.thermal_guard, "stop", side_effect=OSError("thermal cleanup failed")):
            with pytest.raises(
                DaemonError,
                match="pm qos execution failed.*pm qos close failed.*thermal cleanup failed",
            ):
                daemon.start()

        assert daemon.state == DaemonState.STOPPED
        assert not daemon.pid_manager.is_locked
        assert tree.policy_path("policy0", "scaling_min_freq").read_text() == "800000\n"
        assert tree.policy_path("policy4", "scaling_min_freq").read_text() == "800000\n"

    def test_state_snapshots_each_shared_policy_once_with_its_membership(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        manager = StateSnapshotManager(controller, snapshot_file=tmp_path / "snapshot.json")
        plan = controller.build_policy_apply_plan(
            target_khz=4_000_000,
            governor="performance",
            boost=True,
        )

        snapshot = manager.create_snapshot(plan.actions)

        assert set(snapshot.policies) == {"policy0", "policy4"}
        assert snapshot.policies["policy0"].cpus == [0, 2]
        assert snapshot.policies["policy4"].scaling_min_freq == 800_000
        assert snapshot.policies["policy4"].scaling_max_freq is None
        assert snapshot.policies["policy4"].epp is None

    def test_policy_restore_keeps_going_after_one_optional_node_disappears(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        for policy in ("policy0", "policy4"):
            tree.policy_path(policy, "cpb").write_text("0\n")
        controller = SysfsController(tree.root)
        plan = controller.build_policy_apply_plan(
            target_khz=4_000_000,
            governor="performance",
            boost=True,
            cpb=True,
            energy_performance_preference="performance",
        )
        manager = StateSnapshotManager(controller, snapshot_file=tmp_path / "snapshot.json")
        manager.create_snapshot(plan.actions)

        tree.policy_path("policy0", "cpb").unlink()
        tree.policy_path("policy4", "scaling_governor").write_text("performance\n")
        tree.policy_path("policy4", "scaling_min_freq").write_text("4000000\n")
        tree.policy_path("policy4", "cpb").write_text("1\n")
        tree.policy_path("policy4", "energy_performance_preference").write_text("performance\n")

        assert manager.restore()
        assert tree.policy_path("policy4", "scaling_governor").read_text() == "schedutil\n"
        assert tree.policy_path("policy4", "scaling_min_freq").read_text() == "800000\n"
        assert tree.policy_path("policy4", "cpb").read_text() == "0\n"
        assert tree.policy_path("policy4", "energy_performance_preference").read_text() == "balance_performance\n"

    def test_policy_restore_writes_min_before_max_when_lowering_the_bound(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        plan = controller.build_policy_apply_plan(target_khz=4_000_000)
        policy4_max = tree.policy_path("policy4", "scaling_max_freq")
        plan.actions.append(
            PolicyApplyAction(
                "policy4",
                "active_max_frequency",
                policy4_max,
                "4400000",
                "4400000",
            )
        )
        manager = StateSnapshotManager(controller, snapshot_file=tmp_path / "snapshot.json")
        manager.create_snapshot(plan.actions)
        tree.policy_path("policy4", "scaling_min_freq").write_text("4300000\n")
        policy4_max.write_text("4500000\n")
        writes: list[str] = []
        original_write = controller._write_absolute_path

        def record_write(path: Path, value: str) -> None:
            if path.parent.name == "policy4" and path.name.startswith("scaling_"):
                writes.append(path.name)
            original_write(path, value)

        with patch.object(controller, "_write_absolute_path", side_effect=record_write):
            assert manager.restore()

        assert writes[:2] == ["scaling_min_freq", "scaling_max_freq"]

    def test_policy_snapshot_ignores_unknown_actions_and_recovers_from_missing_policies(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        controller = SysfsController(tree.root)
        manager = StateSnapshotManager(controller, snapshot_file=tmp_path / "snapshot.json")
        policy0_min = tree.policy_path("policy0", "scaling_min_freq")
        snapshot = manager.create_snapshot(
            [
                PolicyApplyAction("pm_qos", "device", tmp_path / "dma", "0", "release"),
                PolicyApplyAction("gone", "governor", policy0_min, "performance", "schedutil"),
                PolicyApplyAction("policy0", "active_min_frequency", policy0_min, "1", "invalid"),
            ]
        )

        assert snapshot.policies["policy0"].scaling_min_freq is None
        restored = PolicyStateSnapshot.from_dict(
            snapshot.policies["policy0"].to_dict()
        )
        assert restored.cpus == [0, 2]
        with patch.object(controller, "discover_cpufreq_policies", side_effect=OSError("gone")):
            assert manager.restore(snapshot=snapshot)


class TestPolicyIsolationAndInterfaces:
    def test_adaptive_duty_tracks_samples_and_duty_by_policy(self) -> None:
        controller = AdaptiveDutyController(
            policy_targets={"policy0": 2_400_000, "policy4": 4_000_000},
            min_duty_pct=5.0,
            max_duty_pct=50.0,
            duty_step_pct=5.0,
            initial_duty_pct=20.0,
        )

        policy0_duty = controller.update(policy_id="policy0", current_freq_khz=1_800_000)
        policy4_duty = controller.update(policy_id="policy4", current_freq_khz=4_100_000)
        report = controller.get_policy_report()

        assert policy0_duty > 20.0
        assert policy4_duty < 20.0
        assert report["policy0"]["sample_average_khz"] == 1_800_000
        assert report["policy4"]["sample_average_khz"] == 4_100_000

    def test_auto_and_explicit_target_contracts_are_shared_by_config_and_cli(self) -> None:
        auto = BoostLockConfig.from_dict({"target_frequency_khz": "auto"})
        explicit = BoostLockConfig.from_dict({"target_frequency_khz": 3_600_000})
        parser = build_parser()

        assert auto.to_dict()["target_frequency_khz"] == "auto"
        assert explicit.to_dict()["target_frequency_khz"] == 3_600_000
        assert parser.parse_args(["start", "--target", "auto"]).target == "auto"
        assert parser.parse_args(["bench", "--target", "auto"]).target == "auto"

    def test_ipc_reconfiguration_and_status_expose_resolved_policy_targets(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        config = BoostLockConfig(target_frequency_khz=4_000_000)
        daemon = BoostLockDaemon(config=config, sysfs_root=tree.root, dma_latency_path=tmp_path / "dma")

        response = daemon.handle_ipc_request(
            Request(Command.RECONFIGURE, args={"target_frequency_khz": "auto"})
        )
        status = daemon.get_status()

        assert response.success
        assert response.data["policy_targets"]["policy0"]["effective_target_khz"] == 2_400_000
        assert response.data["policy_targets"]["policy4"]["effective_target_khz"] == 4_400_000
        assert status["policies"]["policy4"]["requested_target"] == "auto"
        assert "policy0" in status["text_status"]

    def test_status_reports_policy_members_controls_and_clamps(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        daemon = BoostLockDaemon(
            config=BoostLockConfig(target_frequency_khz=4_000_000),
            sysfs_root=tree.root,
            dma_latency_path=tmp_path / "dma",
        )

        daemon._apply_sysfs_boost_profile()
        daemon._is_locked_boost = True
        status = daemon.get_status()
        policy0 = status["policies"]["policy0"]
        policy4 = status["policies"]["policy4"]

        assert policy0["member_cpus"] == [0, 2]
        assert policy0["driver"] == "arm-cpufreq"
        assert policy0["requested_target"] == 4_000_000
        assert policy0["effective_target_khz"] == 2_400_000
        assert policy0["clamp_reason"] == "clamped to policy maximum 2400000"
        assert policy0["applied_controls"]["active_min_frequency"] == "2400000"
        assert policy0["skipped_controls"]["cpb"] == "node unavailable"
        assert policy4["effective_target_khz"] == 4_000_000
        assert "policy0" in status["text_status"]

    def test_status_tolerates_subsystem_read_failures(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        daemon = BoostLockDaemon(
            config=BoostLockConfig(target_frequency_khz=4_000_000),
            sysfs_root=tree.root,
            dma_latency_path=tmp_path / "dma",
        )

        with patch.object(daemon.sysfs, "read_all_cpus_state", side_effect=OSError("sysfs")), \
             patch.object(daemon.thermal_guard, "get_status", side_effect=OSError("thermal")), \
             patch.object(daemon.pulse_engine, "get_metrics", side_effect=OSError("pulse")):
            status = daemon.get_status()

        assert status["cpu_states"] == {}
        assert status["thermal"] == {}
        assert status["pulse_engine"] == {}

    def test_status_tolerates_invalid_cpu_and_duty_values(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)
        daemon = BoostLockDaemon(
            config=BoostLockConfig(target_frequency_khz=4_000_000),
            sysfs_root=tree.root,
            dma_latency_path=tmp_path / "dma",
        )
        metrics = daemon.pulse_engine.get_metrics()
        metrics.overall_duty_cycle_pct = "invalid"

        with patch.object(daemon.sysfs, "read_all_cpus_state", return_value={0: None}), \
             patch.object(daemon.pulse_engine, "get_metrics", return_value=metrics):
            status = daemon.get_status()

        assert status["per_cpu"] == {}
        assert status["duty_cycle"] is None

    def test_benchmark_reports_one_result_per_policy_and_its_effective_target(self, tmp_path: Path) -> None:
        tree = _two_policy_tree(tmp_path)

        runner = BenchmarkRunner(target_khz="auto", duration_s=0.01, sysfs_root=tree.root)
        result = runner.resolve_policy_targets()

        assert result["policy0"]["effective_target_khz"] == 2_400_000
        assert result["policy4"]["effective_target_khz"] == 4_400_000

    def test_benchmark_reports_policy_rows_and_handles_unusable_policies(self, tmp_path: Path) -> None:
        child = BenchResult(2_400_000, 0.1, 1, [0], compliance_rate=1.0)
        report = BenchResult(
            4_000_000,
            0.1,
            1,
            [0],
            compliance_rate=1.0,
            policy_results={"policy0": child},
        ).format_report()
        assert "Per-policy compliance" in report

        runner = BenchmarkRunner(target_khz="auto", duration_s=0.01, sysfs_root=tmp_path)
        runner._sysfs = SimpleNamespace(
            discover_cpufreq_policies=lambda: [
                SimpleNamespace(
                    identifier="broken",
                    cpus=[0],
                    driver=None,
                    hardware_min_khz=None,
                    active_min_khz=None,
                    hardware_max_khz=None,
                    active_max_khz=None,
                )
            ],
            get_online_cpus=lambda: (_ for _ in ()).throw(OSError("offline")),
        )

        assert runner.resolve_policy_targets() == {}
        incompatible = SimpleNamespace(
            hardware_min_khz=3_000_000,
            active_min_khz=800_000,
            hardware_max_khz=3_500_000,
            active_max_khz=2_500_000,
        )
        assert runner._effective_policy_target(incompatible)[0] is None
        with pytest.raises(ValueError, match="numeric target"):
            runner._compute_result(None, None, 0.0)

    def test_service_uses_automatic_policy_targets(self) -> None:
        service = (Path(__file__).parents[1] / "boostlock" / "data" / "boostlock.service").read_text()
        exec_start = next(line for line in service.splitlines() if line.startswith("ExecStart="))

        assert "boostlock start" in exec_start
        assert "--target" not in exec_start
