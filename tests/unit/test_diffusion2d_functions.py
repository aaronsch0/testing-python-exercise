"""
Tests for functions in class SolveDiffusion2D
"""

import pytest
import unittest

from diffusion2d import SolveDiffusion2D


# Original pytest-style tests (commented out):
#
# def test_initialize_domain():
#     """
#     Check function SolveDiffusion2D.initialize_domain
#     """
#     solver = SolveDiffusion2D()
#     solver.initialize_domain(w=20.0, h=10.0, dx=0.5, dy=0.5)
#     assert solver.nx == 40
#     assert solver.ny == 20
#
#
# def test_initialize_physical_parameters():
#     """
#     Checks function SolveDiffusion2D.initialize_domain
#     """
#     solver = SolveDiffusion2D()
#     solver.dx = 0.5
#     solver.dy = 0.5
#     solver.initialize_physical_parameters(d=4.0, T_cold=100.0, T_hot=500.0)
#
#     expected = (solver.dx * solver.dx) * (solver.dy * solver.dy) / (
#         2 * 4.0 * (solver.dx * solver.dx + solver.dy * solver.dy)
#     )
#
#     assert solver.dt == pytest.approx(expected)
#
#
# def test_set_initial_condition():
#     """
#     Checks function SolveDiffusion2D.get_initial_function
#     """
#     solver = SolveDiffusion2D()
#     solver.nx = 40
#     solver.ny = 20
#     solver.dx = 0.5
#     solver.dy = 0.5
#     solver.T_cold = 100.0
#     solver.T_hot = 500.0
#     solver.u = solver.set_initial_condition()
#
#     assert hasattr(solver, "u")
#     assert solver.u is not None


class TestDiffusion2D(unittest.TestCase):
    def setUp(self):
        self.solver = SolveDiffusion2D()

    def test_initialize_domain(self):
        self.solver.initialize_domain(w=20.0, h=10.0, dx=0.5, dy=0.5)
        self.assertEqual(self.solver.nx, 40)
        self.assertEqual(self.solver.ny, 20)

    def test_initialize_physical_parameters(self):
        self.solver.dx = 0.5
        self.solver.dy = 0.5
        self.solver.initialize_physical_parameters(d=4.0, T_cold=100.0, T_hot=500.0)

        expected = (
            (self.solver.dx * self.solver.dx) * (self.solver.dy * self.solver.dy)
            / (2 * 4.0 * (self.solver.dx * self.solver.dx + self.solver.dy * self.solver.dy))
        )

        self.assertAlmostEqual(self.solver.dt, expected, places=7)

    def test_set_initial_condition(self):
        self.solver.nx = 40
        self.solver.ny = 20
        self.solver.dx = 0.5
        self.solver.dy = 0.5
        self.solver.T_cold = 100.0
        self.solver.T_hot = 500.0
        self.solver.u = self.solver.set_initial_condition()

        self.assertIsNotNone(self.solver.u)
