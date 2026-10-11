"""Einheiten des IMM-Kerns gegen Handrechnung und Randfälle."""

import numpy as np
import pytest

import imm_algorithm as A
import imm_constants as C
import imm_scenario as S


def test_transition_matrix_rows_sum_to_one_and_stay_probability():
    T = A.transition_matrix(0.2, 2)
    assert T == pytest.approx(np.array([[0.8, 0.2], [0.2, 0.8]]))
    T3 = A.transition_matrix(0.3, 3)
    assert T3.sum(axis=1) == pytest.approx(np.ones(3)) and T3[0, 0] == pytest.approx(0.7) and T3[0, 1] == pytest.approx(0.15)
    assert A.transition_matrix(0.5, 1) == pytest.approx(np.ones((1, 1)))


def test_mixing_hand_example(mix_example):
    xs, Ps, mu = mix_example
    x_mix, P_mix, c = A.imm_mix(xs, Ps, mu, A.transition_matrix(0.2, 2))
    assert c == pytest.approx([0.5, 0.5])
    assert x_mix[0] == pytest.approx([2.0, 0.0]) and x_mix[1] == pytest.approx([8.0, 0.0])
    assert P_mix[0][0, 0] == pytest.approx(17.0) and P_mix[1][0, 0] == pytest.approx(17.0)
    assert P_mix[0][1, 1] == pytest.approx(1.0)


def test_mixing_identical_estimates_changes_nothing():
    xs = [np.array([3.0, 1.0])] * 2
    Ps = [np.diag([4.0, 1.0])] * 2
    x_mix, P_mix, c = A.imm_mix(xs, Ps, np.array([0.3, 0.7]), A.transition_matrix(0.1, 2))
    for j in range(2):
        assert x_mix[j] == pytest.approx(xs[0]) and P_mix[j] == pytest.approx(Ps[0])
    assert c.sum() == pytest.approx(1.0)


def test_combine_hand_example(mix_example):
    xs, Ps, mu = mix_example
    x, P = A.imm_combine(xs, Ps, mu)
    assert x == pytest.approx([5.0, 0.0])
    assert P[0, 0] == pytest.approx(1.0 + 25.0)        # Kovarianz 1 plus Streuung der Mittel (je 5 m Abstand)


def test_gate_threshold_values():
    assert A.gate_threshold(0.99) == pytest.approx(6.6349, abs=1e-3)
    assert A.gate_threshold(0.999) == pytest.approx(10.828, abs=1e-2)


def test_model_probabilities_are_distributions_and_stay_inside_zero_one():
    tr = S.simulate(3, 10.0, C.Q_LOW, 0.1, 6.0)
    run = A.run_imm(tr.z, C.DT, (C.Q_LOW, 3.0), 0.05, 10.0, C.V_START_MEAN, C.V_PRIOR_STD, gate_prob=0.999)
    assert run.mu.shape == (C.T_STEPS, 2) and np.all(run.mu >= 0)
    assert run.mu.sum(axis=1) == pytest.approx(np.ones(C.T_STEPS))


def test_gate_only_rejects_when_both_models_disagree_with_the_measurement():
    z = np.zeros(30)
    z[15] = 400.0
    run = A.run_imm(z, 1.0, (0.3, 3.0), 0.05, 10.0, 0.0, 5.0, gate_prob=0.999)
    assert run.rejected[15] and run.rejected.sum() == 1
    wide = A.run_imm(z, 1.0, (0.3, 3.0), 0.05, 10.0, 0.0, 5.0, gate_prob=None)
    assert not wide.rejected.any()


def test_simulation_is_reproducible_and_maneuver_lowers_the_speed():
    a = S.simulate(7, 10.0, 0.3, 0.05, 0.0)
    b = S.simulate(7, 10.0, 0.3, 0.05, 0.0)
    assert np.array_equal(a.s, b.s) and np.array_equal(a.z, b.z)
    brake = S.simulate(7, 10.0, 0.3, 0.05, 8.0)
    end = C.MANEUVER_START + C.MANEUVER_STEPS
    assert brake.v[end] - a.v[end] == pytest.approx(-8.0, abs=1e-9)


def test_constants_are_consistent():
    assert C.DEFAULT_SWITCH in C.SWITCH_CHOICES and C.DEFAULT_GATE in C.GATE_CHOICES
    assert C.MANEUVER_START + C.MANEUVER_STEPS < C.T_STEPS and C.Q_LOW < C.DEFAULT_Q_HIGH
    assert set(C.METHODS) == set(C.METHOD_LABELS) == set(C.COLORS)
