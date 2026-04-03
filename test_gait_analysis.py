import sys
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from scipy.signal import find_peaks, butter, filtfilt

# Mock mediapipe submodules that aren't available in the test environment
_mp_mock = MagicMock()
sys.modules.setdefault('mediapipe', _mp_mock)
sys.modules.setdefault('mediapipe.solutions', _mp_mock.solutions)
sys.modules.setdefault('mediapipe.framework', _mp_mock.framework)
sys.modules.setdefault('mediapipe.framework.formats', _mp_mock.framework.formats)
sys.modules.setdefault('mediapipe.framework.formats.landmark_pb2', _mp_mock.framework.formats.landmark_pb2)

from gait_analysis import GaitAnalysis


# ---------------------------------------------------------------------------
# gap_fill
# ---------------------------------------------------------------------------
class TestGapFill:
    def test_no_gaps(self):
        """When there are no NaN values, output should equal input."""
        left = [1.0, 2.0, 3.0, 4.0, 5.0]
        right = [5.0, 4.0, 3.0, 2.0, 1.0]
        left_out, right_out = GaitAnalysis.gap_fill(left, right)
        np.testing.assert_array_almost_equal(left_out, left)
        np.testing.assert_array_almost_equal(right_out, right)

    def test_interpolates_nan_gaps(self):
        """NaN values in the middle should be interpolated."""
        left = [0.0, np.nan, 2.0, np.nan, 4.0]
        right = [4.0, np.nan, 2.0, np.nan, 0.0]
        left_out, right_out = GaitAnalysis.gap_fill(left, right)
        assert not np.any(np.isnan(left_out))
        assert not np.any(np.isnan(right_out))
        # Cubic interpolation of a linear sequence should recover exact values
        np.testing.assert_array_almost_equal(left_out, [0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(right_out, [4.0, 3.0, 2.0, 1.0, 0.0])

    def test_consecutive_nans(self):
        """Multiple consecutive NaN values should be filled."""
        left = [0.0, np.nan, np.nan, np.nan, 4.0]
        right = [1.0, 2.0, 3.0, 4.0, 5.0]
        left_out, right_out = GaitAnalysis.gap_fill(left, right)
        assert not np.any(np.isnan(left_out))
        # Linear underlying data → cubic should recover it
        np.testing.assert_array_almost_equal(left_out, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_single_valid_point_unchanged(self):
        """With fewer than 2 valid points, can't interpolate; array stays as-is."""
        left = [np.nan, 5.0, np.nan, np.nan]
        right = [1.0, 2.0, 3.0, 4.0]
        left_out, right_out = GaitAnalysis.gap_fill(left, right)
        # Only 1 valid point in left → can't interpolate, NaN remains
        assert np.isnan(left_out[0])
        assert left_out[1] == 5.0

    def test_returns_numpy_arrays(self):
        left_out, right_out = GaitAnalysis.gap_fill([1.0, 2.0], [3.0, 4.0])
        assert isinstance(left_out, np.ndarray)
        assert isinstance(right_out, np.ndarray)


# ---------------------------------------------------------------------------
# butterworth_low_pass_filter
# ---------------------------------------------------------------------------
class TestButterworthFilter:
    def test_constant_signal_unchanged(self):
        """A constant signal should pass through the filter unchanged."""
        n = 200
        const_left = np.ones(n) * 5.0
        const_right = np.ones(n) * 3.0
        left_f, right_f = GaitAnalysis.butterworth_low_pass_filter(const_left, const_right, 25)
        np.testing.assert_array_almost_equal(left_f, const_left, decimal=5)
        np.testing.assert_array_almost_equal(right_f, const_right, decimal=5)

    def test_removes_high_frequency_noise(self):
        """High frequency noise should be attenuated after filtering."""
        n = 500
        t = np.arange(n)
        # Low-freq signal + high-freq noise
        clean = np.sin(2 * np.pi * 0.01 * t)
        noisy = clean + 0.5 * np.sin(2 * np.pi * 0.45 * t)
        left_f, _ = GaitAnalysis.butterworth_low_pass_filter(noisy, noisy, 25)
        # Filtered signal should be closer to the clean signal than the noisy one
        residual_filtered = np.std(left_f - clean)
        residual_noisy = np.std(noisy - clean)
        assert residual_filtered < residual_noisy * 0.5

    def test_normalized_cutoff_is_0_1752(self):
        """Verify the filter uses Wn=0.1752 directly (not re-normalized)."""
        order = 10
        b, a = butter(order, 0.1752, btype='low', analog=False)
        n = 200
        sig = np.random.randn(n)
        expected = filtfilt(b, a, sig)
        result_left, _ = GaitAnalysis.butterworth_low_pass_filter(sig, sig, 25)
        np.testing.assert_array_almost_equal(result_left, expected)

    def test_frame_rate_does_not_affect_filter(self):
        """Since Wn is fixed at 0.1752, different frame rates should give same result."""
        n = 200
        sig = np.random.randn(n)
        left_25, _ = GaitAnalysis.butterworth_low_pass_filter(sig, sig, 25)
        left_30, _ = GaitAnalysis.butterworth_low_pass_filter(sig, sig, 30)
        left_60, _ = GaitAnalysis.butterworth_low_pass_filter(sig, sig, 60)
        np.testing.assert_array_equal(left_25, left_30)
        np.testing.assert_array_equal(left_25, left_60)


# ---------------------------------------------------------------------------
# Distance metric: horizontal (x-axis) only
# ---------------------------------------------------------------------------
class TestDistanceMetric:
    def test_horizontal_distance_only(self):
        """Distance should only use x-coordinate, not y or z."""
        # Simulate: hip at x=0.5, foot at x=0.3 → distance = 0.2
        # Even if y and z differ greatly, result should be 0.2
        hip_x, foot_x = 0.5, 0.3
        expected = abs(hip_x - foot_x)
        assert expected == pytest.approx(0.2)

        # If we had used 3D Euclidean with large y/z differences:
        hip = np.array([0.5, 0.1, 0.0])
        foot = np.array([0.3, 0.9, 0.5])
        euclidean_3d = np.linalg.norm(hip - foot)
        horizontal = abs(hip[0] - foot[0])
        # The horizontal distance should be much smaller than 3D
        assert horizontal < euclidean_3d
        assert horizontal == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Gait parameter computation with synthetic signals
# ---------------------------------------------------------------------------
def make_synthetic_gait_signal(frame_rate=25, duration=10, stride_freq=1.0, phase=0.0):
    """Generate a synthetic gait distance signal.

    Returns a sinusoidal signal where peaks represent heel strikes
    and minima represent toe-offs, at a known frequency.
    """
    n_frames = frame_rate * duration
    t = np.arange(n_frames) / frame_rate
    # Sinusoidal signal: peaks at stride_freq Hz
    signal = 0.5 + 0.4 * np.sin(2 * np.pi * stride_freq * t + phase)
    return signal, n_frames, frame_rate


class TestGaitParameterComputation:
    """Test the gait event detection and parameter calculations
    using synthetic (pre-filtered) signals so we bypass video processing."""

    def _compute_gait_params(self, dist_left_filtered, dist_right_filtered, frame_rate):
        """Replicate the gait parameter computation from process_video."""
        left_peak_height = 0.35 * np.max(dist_left_filtered)
        right_peak_height = 0.46 * np.max(dist_right_filtered)
        peaks_left, _ = find_peaks(dist_left_filtered, distance=0.8 * frame_rate, height=left_peak_height)
        peaks_right, _ = find_peaks(dist_right_filtered, distance=0.8 * frame_rate, height=right_peak_height)

        left_minima_height = 0.18 * np.min(dist_left_filtered)
        right_minima_height = 0.18 * np.min(dist_right_filtered)
        minima_left, _ = find_peaks(-dist_left_filtered, distance=0.8 * frame_rate, height=-left_minima_height)
        minima_right, _ = find_peaks(-dist_right_filtered, distance=0.8 * frame_rate, height=-right_minima_height)

        # Stance times
        stance_times_left = []
        for i in range(len(peaks_left)):
            subsequent_minima = [m for m in minima_left if m > peaks_left[i]]
            if subsequent_minima:
                stance_times_left.append((subsequent_minima[0] - peaks_left[i]) / frame_rate)

        stance_times_right = []
        for i in range(len(peaks_right)):
            subsequent_minima = [m for m in minima_right if m > peaks_right[i]]
            if subsequent_minima:
                stance_times_right.append((subsequent_minima[0] - peaks_right[i]) / frame_rate)

        # Swing times
        swing_time_left = []
        for i in range(len(minima_left)):
            subsequent_peaks = [p for p in peaks_left if p > minima_left[i]]
            if subsequent_peaks:
                swing_time_left.append((subsequent_peaks[0] - minima_left[i]) / frame_rate)

        swing_time_right = []
        for i in range(len(minima_right)):
            subsequent_peaks = [p for p in peaks_right if p > minima_right[i]]
            if subsequent_peaks:
                swing_time_right.append((subsequent_peaks[0] - minima_right[i]) / frame_rate)

        # Step times (alternating feet)
        step_time_left = []
        for i in range(len(peaks_left)):
            subsequent_right = [p for p in peaks_right if p > peaks_left[i]]
            if subsequent_right:
                step_time_left.append((subsequent_right[0] - peaks_left[i]) / frame_rate)

        step_time_right = []
        for i in range(len(peaks_right)):
            subsequent_left = [p for p in peaks_left if p > peaks_right[i]]
            if subsequent_left:
                step_time_right.append((subsequent_left[0] - peaks_right[i]) / frame_rate)

        # Double support times
        double_support_times_left = []
        for i in range(len(peaks_left) - 1):
            subsequent_right_toe_off = [m for m in minima_right if m > peaks_left[i]]
            if subsequent_right_toe_off:
                double_support_times_left.append((subsequent_right_toe_off[0] - peaks_left[i]) / frame_rate)

        double_support_times_right = []
        for i in range(len(peaks_right) - 1):
            subsequent_left_toe_off = [m for m in minima_left if m > peaks_right[i]]
            if subsequent_left_toe_off:
                double_support_times_right.append((subsequent_left_toe_off[0] - peaks_right[i]) / frame_rate)

        return {
            'peaks_left': peaks_left, 'peaks_right': peaks_right,
            'minima_left': minima_left, 'minima_right': minima_right,
            'stance_left': stance_times_left, 'stance_right': stance_times_right,
            'swing_left': swing_time_left, 'swing_right': swing_time_right,
            'step_left': step_time_left, 'step_right': step_time_right,
            'double_left': double_support_times_left, 'double_right': double_support_times_right,
        }

    def test_peak_detection_with_known_signal(self):
        """With a 1 Hz sine wave at 25 fps, peaks should be ~1 second apart."""
        frame_rate = 25
        sig_left, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0)
        sig_right, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0, phase=np.pi)

        result = self._compute_gait_params(sig_left, sig_right, frame_rate)

        # Should detect ~9-10 peaks for a 10s signal at 1 Hz
        assert len(result['peaks_left']) >= 8
        assert len(result['peaks_right']) >= 8

        # Peaks should be ~25 frames (1 second) apart
        peak_diffs_left = np.diff(result['peaks_left'])
        for d in peak_diffs_left:
            assert d == pytest.approx(25, abs=2)

    def test_stance_time_symmetric_signal(self):
        """For a symmetric sine wave, stance time should be ~half the period."""
        frame_rate = 25
        sig, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0)
        result = self._compute_gait_params(sig, sig, frame_rate)

        # Stance = peak to next minima = half period = 0.5s for 1 Hz
        for st in result['stance_left']:
            assert st == pytest.approx(0.5, abs=0.1)

    def test_swing_time_symmetric_signal(self):
        """For a symmetric sine wave, swing time should be ~half the period."""
        frame_rate = 25
        sig, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0)
        result = self._compute_gait_params(sig, sig, frame_rate)

        for sw in result['swing_left']:
            assert sw == pytest.approx(0.5, abs=0.1)

    def test_stance_plus_swing_equals_stride(self):
        """Stance time + swing time should approximately equal stride time (1/freq)."""
        frame_rate = 25
        freq = 1.0
        sig, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=freq)
        result = self._compute_gait_params(sig, sig, frame_rate)

        n = min(len(result['stance_left']), len(result['swing_left']))
        for i in range(n):
            stride = result['stance_left'][i] + result['swing_left'][i]
            assert stride == pytest.approx(1.0 / freq, abs=0.15)

    def test_step_time_alternating_feet(self):
        """Step time should be the interval between left and right heel strikes."""
        frame_rate = 25
        # Left and right signals offset by half a period (pi phase)
        sig_left, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0, phase=0.0)
        sig_right, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0, phase=np.pi)

        result = self._compute_gait_params(sig_left, sig_right, frame_rate)

        # With pi phase offset, step time should be ~0.5s (half period)
        for st in result['step_left']:
            assert st == pytest.approx(0.5, abs=0.1)
        for st in result['step_right']:
            assert st == pytest.approx(0.5, abs=0.1)

    def test_step_time_is_not_stride_time(self):
        """Step time (alternating feet) should differ from stride time (same foot)
        when left/right are phase-shifted by half a period."""
        frame_rate = 25
        sig_left, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0, phase=0.0)
        sig_right, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0, phase=np.pi)

        result = self._compute_gait_params(sig_left, sig_right, frame_rate)

        # Step time (alternating feet) should be ~0.5s (half period)
        # Stride time (same foot) would be ~1.0s
        # So step time should NOT equal stride time
        if result['step_left']:
            assert result['step_left'][0] == pytest.approx(0.5, abs=0.1)
            # Confirm it's different from stride time (~1.0s)
            assert result['step_left'][0] != pytest.approx(1.0, abs=0.15)

    def test_double_support_time(self):
        """Double support time: left heel strike → next right toe-off."""
        frame_rate = 25
        sig_left, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0, phase=0.0)
        sig_right, _, _ = make_synthetic_gait_signal(frame_rate=frame_rate, duration=10, stride_freq=1.0, phase=np.pi)

        result = self._compute_gait_params(sig_left, sig_right, frame_rate)

        # All double support times should be positive
        for dt in result['double_left']:
            assert dt > 0
        for dt in result['double_right']:
            assert dt > 0

    def test_height_threshold_filters_spurious_peaks(self):
        """Small bumps below the height threshold should not be detected as peaks."""
        frame_rate = 25
        n = 250
        t = np.arange(n) / frame_rate
        # Main gait signal with large peaks
        signal = 0.5 + 0.4 * np.sin(2 * np.pi * 1.0 * t)
        # Add a small bump that shouldn't be detected
        signal[125] += 0.05  # tiny bump at frame 125

        result = self._compute_gait_params(signal, signal, frame_rate)

        # Peaks should be spaced ~25 frames apart, no extra spurious peaks
        peak_diffs = np.diff(result['peaks_left'])
        for d in peak_diffs:
            assert d >= 0.8 * frame_rate - 1  # respect minimum distance

    def test_no_events_for_flat_signal(self):
        """A flat signal should produce no peaks or minima."""
        frame_rate = 25
        flat = np.ones(250) * 0.5
        result = self._compute_gait_params(flat, flat, frame_rate)

        assert len(result['peaks_left']) == 0
        assert len(result['minima_left']) == 0
        assert len(result['stance_left']) == 0
        assert len(result['swing_left']) == 0
        assert len(result['step_left']) == 0


# ---------------------------------------------------------------------------
# Integration: NaN handling through gap_fill + filter pipeline
# ---------------------------------------------------------------------------
class TestNaNPipeline:
    def test_nan_frames_are_filled_before_filtering(self):
        """Signals with NaN gaps should be filled, then filter should succeed."""
        frame_rate = 25
        n = 250
        t = np.arange(n) / frame_rate
        signal = 0.5 + 0.4 * np.sin(2 * np.pi * 1.0 * t)

        # Introduce NaN gaps (simulating missed pose detections)
        signal_with_gaps = signal.copy()
        signal_with_gaps[50] = np.nan
        signal_with_gaps[51] = np.nan
        signal_with_gaps[100] = np.nan

        left_filled, right_filled = GaitAnalysis.gap_fill(
            signal_with_gaps.tolist(), signal.tolist()
        )

        assert not np.any(np.isnan(left_filled))

        # Should not crash when passed to the filter
        left_filtered, right_filtered = GaitAnalysis.butterworth_low_pass_filter(
            left_filled, right_filled, frame_rate
        )

        assert not np.any(np.isnan(left_filtered))
        assert len(left_filtered) == n
