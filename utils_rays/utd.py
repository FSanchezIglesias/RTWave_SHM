"""Uniform Theory of Diffraction (UTD) helpers for edge/corner diffraction.

The ray tracer launches, from every vertex a ray family passes, a fan of
diffracted rays whose amplitude follows the half-plane Fresnel transition
function evaluated at the dominant frequency (Kouyoumjian & Pathak 1974):

* ``transition_T(w)`` is twice the magnitude of the diffracted field relative
  to the incident field, ``T(0) = 1`` (half the GO jump exactly on the shadow
  boundary) and ``T ~ 1/(sqrt(pi) w)`` far from it (Keller's diffraction cone).
* ``fresnel_w(k, rho, dphi)`` is the detour parameter of an observation point
  at distance ``rho`` from the edge and angle ``dphi`` from the boundary.
* ``sommerfeld_half_plane`` is the exact plane-wave half-plane solution used
  by the tests.
"""
import numpy as np
from scipy.special import fresnel

_SQRT_2_OVER_PI = np.sqrt(2. / np.pi)


def fresnel_w(k, rho, dphi):
    """Detour parameter ``w = sqrt(2 k rho) |sin(dphi/2)|``."""
    return np.sqrt(2. * k * np.asarray(rho, dtype=float)) * np.abs(np.sin(0.5 * np.asarray(dphi, dtype=float)))


def transition_T(w):
    """``2 |F(-|w|)| / sqrt(pi)`` with ``F(a) = int_a^inf exp(j tau^2) dtau``.

    Equals ``sqrt(2) * sqrt((1/2 - C(z))^2 + (1/2 - S(z))^2)`` with ``C``/``S``
    the Fresnel integrals of ``z = w sqrt(2/pi)``; 1 at ``w = 0`` and
    ``~ 1/(sqrt(pi) w)`` for large ``w``.
    """
    z = np.abs(np.asarray(w, dtype=float)) * _SQRT_2_OVER_PI
    S, C = fresnel(z)
    return np.sqrt(2.) * np.sqrt((0.5 - C) ** 2 + (0.5 - S) ** 2)


def _phi_fn(w):
    """Complex ``exp(-j pi/4)/sqrt(pi) * int_{-w}^{inf} exp(j tau^2) dtau``.

    Magnitude 1 for ``w -> +inf`` (lit), 1/2 at ``w = 0`` and 0 for ``w -> -inf``.
    """
    w = np.asarray(w, dtype=float)
    z = w * _SQRT_2_OVER_PI
    S, C = fresnel(z)               # int_0^w cos/sin(tau^2) = sqrt(pi/2) * C/S(z)
    ct = np.sqrt(np.pi / 2.) * C
    st = np.sqrt(np.pi / 2.) * S
    F = (np.sqrt(np.pi) / 2.) * np.exp(1j * np.pi / 4.) + ct + 1j * st
    return np.exp(-1j * np.pi / 4.) / np.sqrt(np.pi) * F


def sommerfeld_half_plane(k, rho, phi, phi0, rigid=True, L=None):
    """Exact field of a plane wave diffracted by a half-plane (Sommerfeld).

    The half-plane occupies ``phi = 0`` (edge at the origin, faces at
    ``phi = 0`` and ``phi = 2 pi``); the wave comes from direction ``phi0``
    (``0 < phi0 < pi``).  Returns ``u / U_inc`` (complex) at polar position
    ``(rho, phi)``, ``phi`` in ``(0, 2 pi)``.  ``rigid`` selects the Neumann
    (+) or Dirichlet (-) face condition.  ``L`` optionally replaces ``rho``
    in the Fresnel arguments (UTD distance parameter ``rho X / (rho + X)``
    for a point source at distance ``X``).
    """
    rho = np.asarray(rho, dtype=float)
    phi = np.asarray(phi, dtype=float)
    Lr = rho if L is None else np.asarray(L, dtype=float)
    wi = np.sqrt(2. * k * Lr) * np.cos(0.5 * (phi - phi0))
    wr = np.sqrt(2. * k * Lr) * np.cos(0.5 * (phi + phi0))
    ui = np.exp(-1j * k * rho * np.cos(phi - phi0)) * _phi_fn(wi)
    ur = np.exp(-1j * k * rho * np.cos(phi + phi0)) * _phi_fn(wr)
    return ui + ur if rigid else ui - ur
