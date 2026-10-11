"""Szenario: ein Lieferfahrzeug fährt eine Strecke; GPS misst die Position, ein Teil der Messungen sind Ausreißer, optional bremst es."""

from dataclasses import dataclass

import numpy as np

import imm_algorithm as A
import imm_constants as C


@dataclass(frozen=True)
class Track:
    t: np.ndarray
    s: np.ndarray        # wahre Position in m
    v: np.ndarray        # wahre Geschwindigkeit in m/s
    z: np.ndarray        # GPS-Messung (mit Ausreißern)
    outlier: np.ndarray  # bool je Schritt: diese Messung ist ein Ausreißer


def simulate(seed, sigma_z, q, out_share, dv):
    """Wahre Fahrt plus Messungen. Alle Zufallszahlen kommen in fester Reihenfolge aus einem Generator; die Regler skalieren oder schwellen nur.

    So ändert ein Regler die Spur stetig. Die Messung t = 0 ist nie ein Ausreißer (der Filter muss irgendwo starten).
    """
    rng = np.random.default_rng(seed)
    n_proc = rng.standard_normal((C.T_STEPS, 2))
    n_gps = rng.standard_normal(C.T_STEPS)
    u_out = rng.random(C.T_STEPS)
    n_out = rng.standard_normal(C.T_STEPS)
    n_v0 = float(rng.standard_normal())
    F, Q = A.cv_model(C.DT, q)
    L = np.linalg.cholesky(Q)
    acc = np.zeros(C.T_STEPS)
    acc[C.MANEUVER_START:C.MANEUVER_START + C.MANEUVER_STEPS] = -dv / (C.MANEUVER_STEPS * C.DT)
    s, v = np.zeros(C.T_STEPS), np.zeros(C.T_STEPS)
    x = np.array([0.0, C.V_START_MEAN + C.V_START_STD * n_v0])
    for k in range(C.T_STEPS):
        s[k], v[k] = x
        x = F @ x + L @ n_proc[k] + np.array([0.5 * C.DT ** 2, C.DT]) * acc[k]
    outlier = u_out < out_share
    outlier[0] = False
    z = s + sigma_z * n_gps
    z[outlier] = s[outlier] + C.OUTLIER_STD * n_out[outlier]
    return Track(np.arange(C.T_STEPS) * C.DT, s, v, z, outlier)
