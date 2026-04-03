"""
Unit tests for ISL hardware components:
- ISLAntenna (physical measurement generation with noise and attenuation)
- ISLRouter (network encapsulation, latency, jitter, packet loss)
"""

import numpy as np
import pytest
from mission_sim.core.physics.ids import MicrowaveISLMeasurement
from mission_sim.core.physics.components.sensors.isl_antenna import ISLAntenna
from mission_sim.core.cyber.network.isl_router import ISLRouter
from mission_sim.core.cyber.ids import ISLNetworkFrame


# =============================================================================
# ISLAntenna Tests
# =============================================================================

class TestISLAntenna:
    """Test suite for ISL Antenna physical sensor model."""

    @pytest.fixture
    def antenna(self):
        """Default antenna with moderate noise."""
        return ISLAntenna(
            range_noise_std=0.05,      # 5 cm
            angle_noise_std=5e-5,      # ~10 arcsec
            max_range_m=200000.0,      # 200 km
            reference_range_m=1000.0   # 1 km reference
        )

    @pytest.fixture
    def fixed_seed_antenna(self):
        """Antenna with fixed random seed for deterministic noise tests."""
        np.random.seed(42)
        return ISLAntenna(
            range_noise_std=0.05,
            angle_noise_std=5e-5,
            max_range_m=200000.0,
            reference_range_m=1000.0
        )

    def test_measurement_basic(self, antenna):
        """Test that measurement returns a valid MicrowaveISLMeasurement."""
        rel_pos = np.array([1000.0, 500.0, 200.0])
        current_time = 1234.5

        meas = antenna.measure(rel_pos, current_time)

        assert isinstance(meas, MicrowaveISLMeasurement)
        assert meas.phys_timestamp == current_time
        assert meas.range_m > 0
        assert -np.pi <= meas.azimuth_rad <= np.pi
        assert -np.pi/2 <= meas.elevation_rad <= np.pi/2
        assert 0.0 <= meas.signal_strength <= 1.0

    def test_measurement_true_values(self, fixed_seed_antenna):
        """Test that measurement approximates true values (deterministic seed)."""
        rel_pos = np.array([100.0, 0.0, 0.0])   # purely along x-axis
        meas = fixed_seed_antenna.measure(rel_pos, 0.0)

        # True range = 100 m
        assert meas.range_m == pytest.approx(100.0, abs=0.2)
        # True azimuth = 0 (atan2(0, 100) = 0)
        assert meas.azimuth_rad == pytest.approx(0.0, abs=1e-4)
        # True elevation = 0 (z=0)
        assert meas.elevation_rad == pytest.approx(0.0, abs=1e-4)

    def test_measurement_out_of_range(self, antenna):
        """Test that measurement returns None when target is beyond max_range."""
        # Distance > 200 km
        rel_pos = np.array([250000.0, 0.0, 0.0])
        meas = antenna.measure(rel_pos, 0.0)
        assert meas is None

        # Exactly at max_range (should still measure, but signal may be weak)
        rel_pos = np.array([200000.0, 0.0, 0.0])
        meas = antenna.measure(rel_pos, 0.0)
        assert meas is not None

    def test_measurement_too_close(self, antenna):
        """Test that measurement returns None when target is extremely close."""
        rel_pos = np.array([1e-12, 0.0, 0.0])
        meas = antenna.measure(rel_pos, 0.0)
        assert meas is None

    def test_signal_strength_attenuation(self, antenna):
        """Test signal strength follows inverse-square law with reference."""
        # Within reference range -> signal = 1.0
        rel_pos = np.array([500.0, 0.0, 0.0])
        meas = antenna.measure(rel_pos, 0.0)
        assert meas.signal_strength == 1.0

        # At reference range -> signal = 1.0 (by definition)
        rel_pos = np.array([1000.0, 0.0, 0.0])
        meas = antenna.measure(rel_pos, 0.0)
        assert meas.signal_strength == 1.0

        # Beyond reference -> (ref_range / range)^2
        rel_pos = np.array([2000.0, 0.0, 0.0])
        meas = antenna.measure(rel_pos, 0.0)
        expected = (1000.0 / 2000.0) ** 2
        assert meas.signal_strength == pytest.approx(expected, rel=1e-6)

    def test_measurement_noise_distribution(self, antenna):
        """Test that noise has the specified standard deviation (within tolerance)."""
        np.random.seed(123)
        true_rel_pos = np.array([1000.0, 0.0, 0.0])
        ranges = []
        azimuths = []
        for _ in range(1000):
            meas = antenna.measure(true_rel_pos, 0.0)
            ranges.append(meas.range_m)
            azimuths.append(meas.azimuth_rad)

        ranges = np.array(ranges)
        azimuths = np.array(azimuths)

        # Expected std dev: range_noise_std = 0.05
        assert np.std(ranges) == pytest.approx(0.05, rel=0.15)
        # Expected std dev: angle_noise_std = 5e-5
        assert np.std(azimuths) == pytest.approx(5e-5, rel=0.15)


# =============================================================================
# ISLRouter Tests
# =============================================================================

class TestISLRouter:
    """Test suite for ISL Network Router (Cyber domain)."""

    @pytest.fixture
    def router(self):
        """Default router with moderate latency and 5% packet loss."""
        return ISLRouter(
            base_latency_s=0.05,
            jitter_s=0.01,
            packet_loss_rate=0.05
        )

    @pytest.fixture
    def measurement(self):
        """A generic physical measurement payload."""
        return MicrowaveISLMeasurement(
            phys_timestamp=100.0,
            range_m=150.0,
            azimuth_rad=0.2,
            elevation_rad=0.1,
            signal_strength=0.9
        )

    def test_transmit_basic(self, router, measurement):
        """Test that transmit returns an ISLNetworkFrame with correct fields."""
        current_time = 1000.0
        frame = router.transmit(measurement, "DEP_01", "CHIEF", current_time)

        assert isinstance(frame, ISLNetworkFrame)
        assert frame.payload is measurement
        assert frame.source_id == "DEP_01"
        assert frame.dest_id == "CHIEF"
        assert frame.tx_time == current_time
        # rx_time should be >= current_time (since we clamp negative latency)
        assert frame.rx_time >= current_time
        # and not excessively large (allow jitter up to 3 sigma)
        assert frame.rx_time <= current_time + router.base_latency_s + 3 * router.jitter_s

    def test_transmit_packet_loss(self, router, measurement):
        """Test that packet loss causes transmit to return None."""
        # Set 100% packet loss
        router.packet_loss_rate = 1.0
        frame = router.transmit(measurement, "DEP_01", "CHIEF", 1000.0)
        assert frame is None

        # Set 0% loss, should always return
        router.packet_loss_rate = 0.0
        frame = router.transmit(measurement, "DEP_01", "CHIEF", 1000.0)
        assert frame is not None

    def test_transmit_jitter(self, router, measurement):
        """Test that jitter adds random variation to rx_time."""
        # Disable packet loss for this test
        router.packet_loss_rate = 0.0
        router.jitter_s = 0.05
        router.base_latency_s = 0.05
        current_time = 1000.0

        delays = []
        for _ in range(1000):
            frame = router.transmit(measurement, "A", "B", current_time)
            delays.append(frame.rx_time - current_time)

        delays = np.array(delays)
        # Mean delay should be close to base_latency (jitter zero-mean)
        # Allow larger tolerance due to clamping and finite sample size
        assert np.mean(delays) == pytest.approx(router.base_latency_s, abs=0.02)
        # Standard deviation should be close to jitter_s (allowing for clamping effect)
        assert np.std(delays) == pytest.approx(router.jitter_s, rel=0.3)

    def test_transmit_staleness_methods(self, router, measurement):
        """Test that ISLNetworkFrame's get_age and is_stale work correctly."""
        # Disable packet loss and jitter for deterministic test
        router.packet_loss_rate = 0.0
        router.jitter_s = 0.0
        current_time = 1000.0
        frame = router.transmit(measurement, "A", "B", current_time)

        # Age = current_time - measurement timestamp
        age = frame.get_age(current_time)
        assert age == current_time - measurement.phys_timestamp

        # Stale check
        max_delay = 50.0
        # At current_time (1000), age = 900 > 50 => stale
        assert frame.is_stale(current_time, max_delay) is True
        # At time = 120.0, age = 20 < 50 => not stale
        assert frame.is_stale(120.0, max_delay) is False

    def test_transmit_deterministic_seed(self):
        """Test that using a fixed seed yields reproducible behavior."""
        router1 = ISLRouter(base_latency_s=0.1, jitter_s=0.05, packet_loss_rate=0.0, random_seed=42)
        router2 = ISLRouter(base_latency_s=0.1, jitter_s=0.05, packet_loss_rate=0.0, random_seed=42)

        meas = MicrowaveISLMeasurement(phys_timestamp=0.0, range_m=10.0, azimuth_rad=0.0, elevation_rad=0.0, signal_strength=1.0)
        frame1 = router1.transmit(meas, "A", "B", 0.0)
        frame2 = router2.transmit(meas, "A", "B", 0.0)
        assert frame1.rx_time == frame2.rx_time

    def test_router_repr(self, router):
        """Test string representation."""
        repr_str = repr(router)
        assert "ISLRouter" in repr_str
        assert "Latency=0.05s" in repr_str
        assert "DropRate=5.0%" in repr_str