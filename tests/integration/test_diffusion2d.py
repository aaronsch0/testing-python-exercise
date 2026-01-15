"""
Tests for functionality checks in class SolveDiffusion2D
"""

"""
Integration tests for SolveDiffusion2D
"""

import pytest
import numpy as np
import numpy.testing as npt

from diffusion2d import SolveDiffusion2D


def test_initialize_physical_parameters():
    """Initialize domain and physical parameters, then verify dt."""
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=10.0, h=10.0, dx=0.5, dy=0.5)
    solver.initialize_physical_parameters(d=4.0, T_cold=100.0, T_hot=500.0)

    dx2 = solver.dx * solver.dx
    dy2 = solver.dy * solver.dy
    expected = (dx2 * dy2) / (2 * solver.D * (dx2 + dy2))

    assert solver.dt == pytest.approx(expected)


def test_set_initial_conditions():
    """Initialize domain and physical parameters, then verify initial condition array."""
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=10.0, h=10.0, dx=0.5, dy=0.5)
    solver.initialize_physical_parameters(d=4.0, T_cold=100.0, T_hot=500.0)

    # call method and store result
    u = solver.set_initial_condition()

    # build expected array the same way as in set_initial_condition
    expected_u = solver.T_cold * np.ones((solver.nx, solver.ny))
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = solver.T_hot

    npt.assert_array_equal(u, expected_u)


