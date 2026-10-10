# tests/atmospheric_swing/conftest.py
# -*- coding: utf-8 -*-
"""AS 专题测试公共夹具。"""
import numpy as np
import pytest

CHAR_PERIOD = 4.342 * 86400.0
T_MOON = 27.321661 * 86400.0
T_HALF_2TO1 = 0.5 * T_MOON / CHAR_PERIOD


@pytest.fixture(scope="module")
def crtbp():
    from mission_sim.core.physics.models.gravity.universal_crtbp import UniversalCRTBP
    return UniversalCRTBP.earth_moon_system()


@pytest.fixture(scope="module")
def known_2to1_state():
    return np.array([0.8087346540539091, 0.0, 0.0, 0.0, 0.515823613104834, 0.0])
