"""Kalman-Filter, Huber-Filter und IMM (Interacting Multiple Model), numpy-only. Jede Funktion ist eine Einheit mit expliziten Ein- und Ausgaben.

Modell: Zustand x = (Position s, Geschwindigkeit v), konstante Geschwindigkeit mit zufälligem Beschleunigungsrauschen q; GPS misst s.
Das IMM rechnet mehrere solche Modelle mit verschiedenem q parallel und gewichtet sie je Sekunde nach ihrer Messwahrscheinlichkeit.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

H_GPS = np.array([1.0, 0.0])


def cv_model(dt, q):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q ** 2 * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
    return F, Q


def predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q


def innovation(x, P, z, r):
    """Innovation, ihre Varianz S und das normierte Quadrat NIS."""
    S = float(H_GPS @ P @ H_GPS + r)
    nu = float(z - H_GPS @ x)
    return nu, S, nu * nu / S


def update_scalar(x, P, z, r):
    """Kalman-Korrektur (Joseph-Form) mit der GPS-Messung (Varianz r)."""
    S = float(H_GPS @ P @ H_GPS + r)
    K = (P @ H_GPS) / S
    nu = float(z - H_GPS @ x)
    I_KH = np.eye(len(x)) - np.outer(K, H_GPS)
    return x + K * nu, I_KH @ P @ I_KH.T + np.outer(K, K) * r


def gate_threshold(prob):
    """Chi-Quadrat-Quantil mit 1 Freiheitsgrad = (Normalquantil zu (1+prob)/2)^2."""
    return NormalDist().inv_cdf((1.0 + prob) / 2.0) ** 2


def transition_matrix(p_switch, n_models):
    """Modellwechsel-Matrix: bleibt mit 1 - p, wechselt gleichverteilt zu den anderen Modellen."""
    if n_models == 1:
        return np.ones((1, 1))
    off = p_switch / (n_models - 1)
    return np.full((n_models, n_models), off) + np.eye(n_models) * (1.0 - p_switch - off)


def imm_mix(xs, Ps, mu, tpm):
    """Mischen: je Zielmodell j die Anfangswerte als gewichteter Mittelwert aller Modelle (Gewichte aus Modellwahrscheinlichkeit und Wechselmatrix).

    Gibt (x_j, P_j je Modell, c_j = vorhergesagte Modellwahrscheinlichkeit) zurück.
    """
    m = len(xs)
    c = tpm.T @ mu
    w = (tpm * mu[:, None]) / c[None, :]          # w[i, j] = P(Modell i bei k-1 | Modell j bei k)
    x_mix, P_mix = [], []
    for j in range(m):
        xj = sum(w[i, j] * xs[i] for i in range(m))
        Pj = np.zeros((2, 2))
        for i in range(m):
            d = (xs[i] - xj)[:, None]
            Pj += w[i, j] * (Ps[i] + d @ d.T)
        x_mix.append(xj)
        P_mix.append(Pj)
    return x_mix, P_mix, c


def imm_combine(xs, Ps, mu):
    """Gesamtschätzung: gewichteter Mittelwert und Kovarianz (mit Streuung der Modellmittel)."""
    x = sum(mu[j] * xs[j] for j in range(len(xs)))
    P = np.zeros((2, 2))
    for j in range(len(xs)):
        d = (xs[j] - x)[:, None]
        P += mu[j] * (Ps[j] + d @ d.T)
    return x, P


@dataclass(frozen=True)
class Run:
    x: np.ndarray          # (T, 2)
    P: np.ndarray          # (T, 2, 2)
    mu: np.ndarray         # (T, m) Modellwahrscheinlichkeiten (bei Einzelmodellen m = 1)
    rejected: np.ndarray   # (T,) Messung vom Tor verworfen


def run_kf(z, dt, q, sigma_z, v_prior_mean, v_prior_std, huber_c=None):
    """Ein Modell, optional mit Huber-Gewichtung (Varianz wird bei großer Abweichung vergrößert)."""
    T = len(z)
    F, Q = cv_model(dt, q)
    r = sigma_z ** 2
    x = np.array([z[0], v_prior_mean], float)
    P = np.diag([r, v_prior_std ** 2])
    xs, Ps = np.zeros((T, 2)), np.zeros((T, 2, 2))
    for t in range(T):
        if t > 0:
            x, P = predict(x, P, F, Q)
            rr = r
            if huber_c is not None:
                _, _, nis = innovation(x, P, z[t], r)
                rr = r * max(1.0, float(np.sqrt(nis)) / huber_c)
            x, P = update_scalar(x, P, z[t], rr)
        xs[t], Ps[t] = x, P
    return Run(xs, Ps, np.ones((T, 1)), np.zeros(T, bool))


def run_imm(z, dt, qs, p_switch, sigma_z, v_prior_mean, v_prior_std, gate_prob=None):
    """IMM über eine Spur. qs: Beschleunigungsrauschen der Modelle. Mit gate_prob: Messung wird verworfen, wenn sie für ALLE Modelle außerhalb des Tors liegt."""
    T, m = len(z), len(qs)
    models = [cv_model(dt, q) for q in qs]
    tpm = transition_matrix(p_switch, m)
    r = sigma_z ** 2
    thr = gate_threshold(gate_prob) if gate_prob is not None else None
    x0 = np.array([z[0], v_prior_mean], float)
    P0 = np.diag([r, v_prior_std ** 2])
    xs, Ps = [x0.copy() for _ in range(m)], [P0.copy() for _ in range(m)]
    mu = np.full(m, 1.0 / m)
    out_x, out_P = np.zeros((T, 2)), np.zeros((T, 2, 2))
    out_mu, rej = np.zeros((T, m)), np.zeros(T, bool)
    for t in range(T):
        if t > 0:
            xm, Pm, c = imm_mix(xs, Ps, mu, tpm)
            pred = [predict(xm[j], Pm[j], *models[j]) for j in range(m)]
            xs, Ps = [p[0] for p in pred], [p[1] for p in pred]
            nis = np.array([innovation(xs[j], Ps[j], z[t], r)[2] for j in range(m)])
            if thr is not None and nis.min() > thr:
                rej[t] = True
                mu = c
            else:
                lik = np.zeros(m)
                for j in range(m):
                    S = float(H_GPS @ Ps[j] @ H_GPS + r)
                    nu = float(z[t] - H_GPS @ xs[j])
                    lik[j] = np.exp(-0.5 * nu * nu / S) / np.sqrt(2 * np.pi * S)
                    xs[j], Ps[j] = update_scalar(xs[j], Ps[j], z[t], r)
                mu = lik * c
                s = mu.sum()
                mu = mu / s if s > 0 else c
        out_x[t], out_P[t] = imm_combine(xs, Ps, mu)
        out_mu[t] = mu
    return Run(out_x, out_P, out_mu, rej)
