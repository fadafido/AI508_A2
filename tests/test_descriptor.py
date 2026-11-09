"""Known-answer tests for the descriptor maths (src/featuriser/descriptor.py).

The values are computed by hand. The series [2, 4, 6] walks through every step:
    min-max            [0, 0.5, 1]
    z-score (pop. std) std = sqrt(1/6), so [-sqrt(3/2), 0, sqrt(3/2)]
    Cusum              [-sqrt(3/2), -sqrt(3/2), 0]
    final min-max      [0, 0, 1]
    degree-2 fit       through (0,0), (0.5,0), (1,1): 2x^2 - x, so (2, -1, 0) with MAE 0
Two-point series give the min-norm solutions stored in the authors' own descriptor file (D-03):
decreasing (-0.5, -0.5, 1, 0), increasing (0.5, 0.5, 0, 0).

    pytest tests/test_descriptor.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.featuriser.descriptor import garralda_feature, minmax, prats_descriptor  # noqa: E402

S32 = np.sqrt(3.0 / 2.0)


def test_minmax():
    np.testing.assert_allclose(minmax(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])


def test_zscore_step():
    t = np.array([0.0, 0.5, 1.0])
    z = (t - t.mean()) / t.std()
    np.testing.assert_allclose(z, [-S32, 0.0, S32], atol=1e-12)


def test_cusum_step():
    z = np.array([-S32, 0.0, S32])
    np.testing.assert_allclose(np.cumsum(z), [-S32, -S32, 0.0], atol=1e-12)


def test_final_minmax_step():
    np.testing.assert_allclose(minmax(np.array([-S32, -S32, 0.0])), [0.0, 0.0, 1.0], atol=1e-12)


def test_degree2_fit():
    # the parabola through (0,0), (0.5,0), (1,1) is 2x^2 - x
    np.testing.assert_allclose(garralda_feature([2.0, 4.0, 6.0]), [2.0, -1.0, 0.0, 0.0], atol=1e-9)


def test_two_point_series_match_the_authors_file():
    np.testing.assert_allclose(garralda_feature([5.0, 3.0]), [-0.5, -0.5, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(garralda_feature([3.0, 5.0]), [0.5, 0.5, 0.0, 0.0], atol=1e-12)


def test_constant_series_is_zero():
    np.testing.assert_allclose(garralda_feature([7.0, 7.0, 7.0]), [0.0, 0.0, 0.0, 0.0])


def test_scale_and_shift_invariance():
    # min-max then z-score make the fit invariant to scaling and shifting the series
    a = garralda_feature([1.0, 4.0, 9.0, 2.0])
    b = garralda_feature([10.0, 40.0, 90.0, 20.0])
    c = garralda_feature([101.0, 104.0, 109.0, 102.0])
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(a, c, atol=1e-9)


def test_prats_descriptor_known_answer():
    m = np.zeros((2, 24))                          # 2 tasks x 24 metrics
    m[:, 0] = [1.0, 3.0]
    m[:, 1] = [10.0, 10.0]
    tps = np.array([2.0])
    v = prats_descriptor(m, tps)
    # mean, min, max, population std per column, then tasks-per-stage
    np.testing.assert_allclose(v[:4], [2.0, 1.0, 3.0, 1.0])
    np.testing.assert_allclose(v[4:8], [10.0, 10.0, 10.0, 0.0])
    np.testing.assert_allclose(v[-4:], [2.0, 2.0, 2.0, 0.0])
