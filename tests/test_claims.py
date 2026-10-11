"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (40 feste Fahrten).

Plattformrobust: Bänder statt exakter Werte (CI installiert neueste numpy/scipy)."""

import numpy as np
import pytest

import imm_constants as C
import imm_evaluation as ev


@pytest.fixture(scope="module")
def maneuver_rows():
    return {r["dv"]: r for r in ev.sweep_maneuver(ev.Settings())}


@pytest.fixture(scope="module")
def mode_result():
    return ev.sweep_mode_probability(ev.Settings())


@pytest.fixture(scope="module")
def switch_rows():
    return {r["p"]: r for r in ev.sweep_switch(ev.Settings())}


def test_imm_error_is_flat_in_the_maneuver_while_single_models_grow(maneuver_rows):
    clean0, clean10 = maneuver_rows[0.0]["clean"], maneuver_rows[10.0]["clean"]
    assert clean0["kf_low"]["rmse"] == pytest.approx(4.5, abs=0.5)
    assert clean10["kf_low"]["rmse"] == pytest.approx(8.5, abs=0.8)
    assert clean10["huber"]["rmse"] == pytest.approx(9.8, abs=0.9)
    assert clean0["imm_gate"]["rmse"] == pytest.approx(5.3, abs=0.5)
    assert clean10["imm_gate"]["rmse"] == pytest.approx(5.8, abs=0.6)
    assert clean10["imm_gate"]["rmse"] < clean10["kf_low"]["rmse"] - 2.0
    assert clean10["imm_gate"]["rmse"] - clean0["imm_gate"]["rmse"] < 1.0


def test_two_models_cost_accuracy_without_a_maneuver(maneuver_rows):
    c0 = maneuver_rows[0.0]["clean"]
    assert c0["imm"]["rmse"] > c0["kf_low"]["rmse"] * 1.1
    assert c0["kf_high"]["rmse"] > c0["kf_low"]["rmse"] * 1.3
    d0 = maneuver_rows[0.0]["dirty"]
    assert d0["huber"]["rmse"] < d0["imm_gate"]["rmse"]               # ohne Manöver ist Huber mit Ausreißern besser


def test_with_outliers_and_a_strong_maneuver_the_gated_imm_is_best_and_the_ungated_one_fails(maneuver_rows):
    d6, d10 = maneuver_rows[6.0]["dirty"], maneuver_rows[10.0]["dirty"]
    assert d6["imm_gate"]["rmse"] == pytest.approx(6.7, abs=0.7) and d6["huber"]["rmse"] == pytest.approx(7.1, abs=0.8)
    assert d10["imm_gate"]["rmse"] < d10["huber"]["rmse"] - 2.5
    for dv in C.DV_SWEEP:
        d = maneuver_rows[dv]["dirty"]
        assert d["imm"]["rmse"] > 1.7 * d["imm_gate"]["rmse"]
        assert d["imm"]["rmse"] == pytest.approx(13.7, abs=1.5)
        assert d["imm"]["lost"] == pytest.approx(0.025, abs=0.04) and d["imm_gate"]["lost"] == 0.0


def test_detection_is_slow_and_the_gate_keeps_probabilities_clean(mode_result):
    clean, dirty = mode_result["clean"], mode_result["dirty"]
    pre = slice(20, C.MANEUVER_START - 2)
    assert float(clean["mu_gate"][pre].mean()) == pytest.approx(0.31, abs=0.06)
    assert float(dirty["mu"][pre].mean()) == pytest.approx(0.48, abs=0.07)
    assert float(dirty["mu_gate"][pre].mean()) < float(dirty["mu"][pre].mean()) - 0.1
    assert clean["detected"] > 0.9 and clean["median_delay"] == pytest.approx(8.0, abs=2.0)
    peak = float(clean["mu_gate"][C.MANEUVER_START:C.MANEUVER_START + 25].max())
    assert peak == pytest.approx(0.57, abs=0.08) and peak > float(clean["mu_gate"][pre].mean()) + 0.2     # steigt nur bis knapp über 50 %


def test_switch_probability_matters_little(switch_rows):
    gated = [r["imm_gate"]["rmse"] for r in switch_rows.values()]
    plain = [r["imm"]["rmse"] for r in switch_rows.values()]
    assert max(gated) - min(gated) < 0.8 and max(plain) - min(plain) < 0.6
    assert np.median(gated) == pytest.approx(6.6, abs=0.8) and np.median(plain) == pytest.approx(13.7, abs=1.5)


def test_sweeps_use_the_fixed_instances():
    assert len(C.SWEEP_SEEDS) == 40 and C.SWEEP_SEEDS[0] == 100000
