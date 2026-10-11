"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import imm_constants as C
import imm_evaluation as ev
import imm_presets as P


def _settings_from(p):
    return ev.Settings(sigma_z=p["sz"], out_share=p["out"], dv=p["dv"], q_high=p["qh"], p_switch=p["sw"], gate=p["gate"], seed=p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert C.SIGMA_Z_MIN <= p["sz"] <= C.SIGMA_Z_MAX and C.OUT_MIN <= p["out"] <= C.OUT_MAX, name
        assert C.DV_MIN <= p["dv"] <= C.DV_MAX and C.Q_HIGH_MIN <= p["qh"] <= C.Q_HIGH_MAX, name
        assert p["sw"] in C.SWITCH_CHOICES and p["gate"] in C.GATE_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standard (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers_and_snaps_choices():
    out = P.SETTING_SPECS["out_slider"]
    assert P.clamp_value(out, 5.0) == C.OUT_MAX and P.clamp_value(out, -1.0) == C.OUT_MIN
    sw = P.SETTING_SPECS["switch_select"]
    assert P.clamp_value(sw, 0.05) == 0.05 and P.clamp_value(sw, 0.07) is None


def test_the_presets_show_what_their_help_text_claims():
    assert C.PRESETS["Nur Manöver, keine Ausreißer"]["out"] == 0.0 and C.PRESETS["Nur Manöver, keine Ausreißer"]["dv"] > 0
    assert C.PRESETS["Viele Ausreißer"]["out"] > C.PRESETS["Standard (Voreinstellung)"]["out"]
    assert C.PRESETS["Gleichmäßige Fahrt"]["dv"] == 0.0
