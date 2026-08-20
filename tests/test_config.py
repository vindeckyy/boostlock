import json
import os
import tempfile
import unittest

from boostlock.config import BoostLockConfig, ConfigValidationError


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = BoostLockConfig()
        cfg.validate()
        self.assertEqual(cfg.target_frequency_khz, 4000000)
        self.assertEqual(cfg.min_pulse_duty_pct, 5.0)
        self.assertEqual(cfg.max_pulse_duty_pct, 50.0)
        self.assertEqual(cfg.duty_step_pct, 2.0)
        self.assertEqual(cfg.thermal_limit_c, 100.0)
        self.assertEqual(cfg.thermal_warn_c, 90.0)
        self.assertEqual(cfg.thermal_recover_c, 85.0)
        self.assertEqual(cfg.poll_interval_ms, 100)
        self.assertEqual(cfg.dma_latency_us, 0)
        self.assertEqual(cfg.governor, "performance")

    def test_custom_valid_config(self):
        cfg = BoostLockConfig(
            target_frequency_khz=3800000,
            min_pulse_duty_pct=10.0,
            max_pulse_duty_pct=60.0,
            duty_step_pct=5.0,
            thermal_limit_c=90.0,
            thermal_warn_c=80.0,
            thermal_recover_c=75.0,
            poll_interval_ms=50,
            dma_latency_us=10,
            governor="performance",
        )
        cfg.validate()
        self.assertEqual(cfg.target_frequency_khz, 3800000)
        self.assertEqual(cfg.thermal_limit_c, 90.0)

    def test_automatic_target_is_valid(self):
        cfg = BoostLockConfig(target_frequency_khz="auto")
        cfg.validate()
        self.assertEqual(cfg.to_dict()["target_frequency_khz"], "auto")

    def test_invalid_target_frequency(self):
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(target_frequency_khz=0).validate()

        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(target_frequency_khz=-1000).validate()

        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(target_frequency_khz=20_000_000).validate()

    def test_invalid_thermal_bounds(self):
        # Thermal limit too low
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(thermal_limit_c=40.0).validate()

        # Thermal limit too high
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(thermal_limit_c=120.0).validate()

        # Thermal warn >= thermal limit
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(thermal_warn_c=85.0, thermal_limit_c=85.0).validate()

        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(thermal_warn_c=90.0, thermal_limit_c=85.0).validate()

        # Thermal recover >= thermal warn
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(thermal_recover_c=75.0, thermal_warn_c=75.0).validate()

        # Thermal recover too low
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(thermal_recover_c=10.0).validate()

    def test_invalid_duty_cycles(self):
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(min_pulse_duty_pct=-1.0).validate()

        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(max_pulse_duty_pct=105.0).validate()

        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(min_pulse_duty_pct=60.0, max_pulse_duty_pct=40.0).validate()

        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(duty_step_pct=0.0).validate()

    def test_invalid_intervals_and_latency(self):
        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(poll_interval_ms=0).validate()

        with self.assertRaises(ConfigValidationError):
            BoostLockConfig(dma_latency_us=-5).validate()

    def test_to_dict_and_from_dict(self):
        cfg = BoostLockConfig(target_frequency_khz=4200000, thermal_limit_c=95.0, thermal_warn_c=85.0, thermal_recover_c=80.0)
        data = cfg.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["target_frequency_khz"], 4200000)
        self.assertEqual(data["thermal_limit_c"], 95.0)

        restored = BoostLockConfig.from_dict(data)
        self.assertEqual(restored.target_frequency_khz, 4200000)
        self.assertEqual(restored.thermal_limit_c, 95.0)
        restored.validate()

    def test_json_serialization_and_file_io(self):
        cfg = BoostLockConfig(target_frequency_khz=4500000)
        json_str = cfg.to_json()
        self.assertIn('"target_frequency_khz": 4500000', json_str)

        restored = BoostLockConfig.from_json(json_str)
        self.assertEqual(restored.target_frequency_khz, 4500000)

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            cfg.to_json(temp_path)
            loaded = BoostLockConfig.from_json(temp_path)
            self.assertEqual(loaded.target_frequency_khz, 4500000)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_from_dict_extra_and_missing_keys(self):
        # Missing keys fallback to defaults
        cfg = BoostLockConfig.from_dict({"target_frequency_khz": 3600000})
        self.assertEqual(cfg.target_frequency_khz, 3600000)
        self.assertEqual(cfg.thermal_limit_c, 100.0)

        # Extra unknown keys are ignored without error
        cfg2 = BoostLockConfig.from_dict({"target_frequency_khz": 3600000, "unknown_field": 123})
        self.assertEqual(cfg2.target_frequency_khz, 3600000)


if __name__ == "__main__":
    unittest.main()
