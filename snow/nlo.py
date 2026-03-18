# -*- coding: utf-8 -*-
"""
Nonlinear Envelope Equation solver.

This module re-exports the SciPy-based solver for backward compatibility.
For GPU acceleration, use snow.nlo_jax directly or pass backend='jax'
to waveguide.propagate_NEE().
"""
from .nlo_scipy import NEE

__all__ = ['NEE']
