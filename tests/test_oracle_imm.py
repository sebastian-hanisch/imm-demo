"""Orakel-Test imm-demo: das IMM gegen ANDERE Rechenwege.

1. Ein Modell bzw. zwei gleiche Modelle: das IMM ist der gewöhnliche Kalman-Filter,
2. exakte Gauß-Mischung über ALLE Modellfolgen (2^k Filter) gegen Modellwahrscheinlichkeiten und Mittelwert des IMM,
3. Huber-Lauf mit unendlicher Schwelle gleich dem Kalman-Filter,
4. die angezeigten Kennzahlen gegen unabhängige Neuberechnung.
"""

import itertools

import numpy as np
import pytest

import imm_algorithm as A
import imm_constants as C
import imm_evaluation as ev
import imm_scenario as S


def test_single_model_imm_equals_the_kalman_filter():
    for seed in range(4):
        tr = S.simulate(seed, 10.0, C.Q_LOW, 0.0, 4.0)
        kf = A.run_kf(tr.z, C.DT, C.Q_LOW, 10.0, C.V_START_MEAN, C.V_PRIOR_STD)
        imm = A.run_imm(tr.z, C.DT, (C.Q_LOW,), 0.1, 10.0, C.V_START_MEAN, C.V_PRIOR_STD)
        assert imm.x == pytest.approx(kf.x, abs=1e-9) and imm.P == pytest.approx(kf.P, abs=1e-9)


def test_two_identical_models_are_the_kalman_filter_and_keep_equal_probabilities():
    tr = S.simulate(2, 10.0, C.Q_LOW, 0.05, 6.0)
    kf = A.run_kf(tr.z, C.DT, 0.7, 10.0, C.V_START_MEAN, C.V_PRIOR_STD)
    imm = A.run_imm(tr.z, C.DT, (0.7, 0.7), 0.1, 10.0, C.V_START_MEAN, C.V_PRIOR_STD)
    assert imm.x == pytest.approx(kf.x, abs=1e-8) and imm.mu == pytest.approx(np.full((C.T_STEPS, 2), 0.5), abs=1e-9)


def _exact_mixture(z, qs, p_switch, sigma_z):
    """Exakter Posterior über ALLE Modellfolgen: je Folge ein Kalman-Filter, Gewicht = Produkt der Wechselwahrscheinlichkeiten mal Likelihood. Gibt je Zeit (mu, x_mittel)."""
    T, m = len(z), len(qs)
    tpm = A.transition_matrix(p_switch, m)
    r = sigma_z ** 2
    models = [A.cv_model(1.0, q) for q in qs]
    x0 = np.array([z[0], C.V_START_MEAN])
    P0 = np.diag([r, C.V_PRIOR_STD ** 2])
    # Hypothesen: (letztes Modell, Gewicht, x, P); Start: gleichverteilt über das Modell zum Zeitpunkt 0, keine Korrektur bei t = 0
    hyps = [(j, 1.0 / m, x0.copy(), P0.copy()) for j in range(m)]
    out = [(np.full(m, 1.0 / m), x0.copy())]
    for t in range(1, T):
        new = []
        for (j_prev, w, x, P) in hyps:
            for j in range(m):
                xp, Pp = A.predict(x, P, *models[j])
                S_ = float(A.H_GPS @ Pp @ A.H_GPS + r)
                nu = float(z[t] - A.H_GPS @ xp)
                lik = np.exp(-0.5 * nu * nu / S_) / np.sqrt(2 * np.pi * S_)
                xu, Pu = A.update_scalar(xp, Pp, z[t], r)
                new.append((j, w * tpm[j_prev, j] * lik, xu, Pu))
        tot = sum(h[1] for h in new)
        hyps = [(j, w / tot, x, P) for j, w, x, P in new]
        mu = np.zeros(m)
        for j, w, x, P in hyps:
            mu[j] += w
        xm = sum(w * x for _, w, x, _ in hyps)
        out.append((mu, xm))
    return out


def test_imm_close_to_the_exact_mixture_over_all_model_sequences():
    qs = (0.3, 3.0)
    worst_mu, worst_x = 0.0, 0.0
    for seed in range(6):
        tr = S.simulate(seed, 10.0, C.Q_LOW, 0.0, 6.0)
        z = tr.z[C.MANEUVER_START - 4:C.MANEUVER_START + 6].copy()       # 10 Sekunden um den Manöverbeginn, 2^9 Folgen
        exact = _exact_mixture(z, qs, 0.05, 10.0)
        imm = A.run_imm(z, 1.0, qs, 0.05, 10.0, C.V_START_MEAN, C.V_PRIOR_STD)
        for t in range(1, len(z)):
            worst_mu = max(worst_mu, abs(imm.mu[t, 1] - exact[t][0][1]))
            worst_x = max(worst_x, abs(imm.x[t, 0] - exact[t][1][0]))
    assert worst_mu < 0.03            # das IMM ist eine Näherung: kleine, aber endliche Abweichung (gemessen etwa 0,012)
    assert worst_x < 0.8


def test_huber_with_infinite_threshold_is_the_kalman_filter():
    tr = S.simulate(5, 10.0, C.Q_LOW, 0.1, 4.0)
    kf = A.run_kf(tr.z, C.DT, C.Q_LOW, 10.0, C.V_START_MEAN, C.V_PRIOR_STD)
    hub = A.run_kf(tr.z, C.DT, C.Q_LOW, 10.0, C.V_START_MEAN, C.V_PRIOR_STD, huber_c=1e9)
    assert hub.x == pytest.approx(kf.x)


def test_displayed_numbers_match_independent_recomputation():
    for seed in (3, 11, 40):
        s = ev.Settings(seed=seed, dv=8.0)
        a = ev.analyse(s)
        tr = S.simulate(seed, s.sigma_z, C.Q_LOW, s.out_share, s.dv)
        imm = A.run_imm(tr.z, C.DT, (C.Q_LOW, s.q_high), s.p_switch, s.sigma_z, C.V_START_MEAN, C.V_PRIOR_STD, gate_prob=s.gate)
        sl = slice(C.N_BURN, None)
        assert a.rmse["imm_gate"] == pytest.approx(np.sqrt(np.mean((imm.x[sl, 0] - tr.s[sl]) ** 2)))
        assert a.lost["imm_gate"] == (abs(imm.x[-1, 0] - tr.s[-1]) > C.FAIL_ERR)
        after = np.flatnonzero(imm.mu[C.MANEUVER_START:, 1] > C.MODE_THRESHOLD)
        assert a.detect_delay_gate == (int(after[0]) if len(after) else None)
