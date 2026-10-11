"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Zeitachse: ein Schritt = eine Sekunde, 120 Schritte
DT = 1.0
T_STEPS = 120
N_BURN = 10                       # Einschwingzeit: Kennzahlen zählen erst ab diesem Schritt

# feste Annahmen (gleiche Fahrt wie im Stück „Robuster Filter“)
V_START_MEAN = 10.0
V_START_STD = 1.0
V_PRIOR_STD = 5.0
OUTLIER_STD = 80.0                # Streuung eines Ausreißers in m
MANEUVER_START = 60               # Beginn des Manövers (Schritt)
MANEUVER_STEPS = 5                # Dauer des Manövers in Schritten
Q_LOW = 0.3                       # ruhiges Modell = wahre Fahrunruhe der Fahrt
HUBER_C = 1.96
FAIL_ERR = 50.0                   # Fahrt gilt als „verloren“, wenn der Endfehler darüber liegt (m)
MODE_THRESHOLD = 0.5              # Manöver gilt als erkannt, wenn die Wahrscheinlichkeit des unruhigen Modells darüber liegt

# Reglergrenzen
SIGMA_Z_MIN, SIGMA_Z_MAX, DEFAULT_SIGMA_Z = 3.0, 30.0, 10.0
OUT_MIN, OUT_MAX, DEFAULT_OUT = 0.0, 0.3, 0.05
DV_MIN, DV_MAX, DEFAULT_DV = 0.0, 10.0, 6.0
Q_HIGH_MIN, Q_HIGH_MAX, DEFAULT_Q_HIGH = 1.0, 6.0, 3.0
SWITCH_CHOICES = (0.005, 0.02, 0.05, 0.1, 0.2)
DEFAULT_SWITCH = 0.05
GATE_CHOICES = (0.99, 0.999, 0.9999)
DEFAULT_GATE = 0.999
SEED_MAX, DEFAULT_SEED = 999999, 35

# feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_SEEDS = tuple(range(100000, 100040))
DV_SWEEP = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
SWITCH_SWEEP = SWITCH_CHOICES

METHODS = ("kf_low", "kf_high", "huber", "imm", "imm_gate")
METHOD_LABELS = {
    "kf_low": "Ein Modell, ruhig",
    "kf_high": "Ein Modell, unruhig",
    "huber": "Ein Modell, ruhig + Huber",
    "imm": "Zwei Modelle (IMM)",
    "imm_gate": "Zwei Modelle + Tor",
}

COLOR_TRUTH = "#2F6B65"
COLORS = {"kf_low": "#7a7a7a", "kf_high": "#8a5a9a", "huber": "#4c78a8", "imm": "#c97b2a", "imm_gate": "#b04a4a"}
COLOR_MODE = "#c97b2a"

PRESETS = {
    "Standard (Voreinstellung)": {"sz": DEFAULT_SIGMA_Z, "out": DEFAULT_OUT, "dv": DEFAULT_DV, "qh": DEFAULT_Q_HIGH, "sw": DEFAULT_SWITCH, "gate": DEFAULT_GATE, "seed": DEFAULT_SEED},
    "Nur Manöver, keine Ausreißer": {"sz": DEFAULT_SIGMA_Z, "out": 0.0, "dv": 8.0, "qh": DEFAULT_Q_HIGH, "sw": DEFAULT_SWITCH, "gate": DEFAULT_GATE, "seed": DEFAULT_SEED},
    "Viele Ausreißer": {"sz": DEFAULT_SIGMA_Z, "out": 0.2, "dv": 6.0, "qh": DEFAULT_Q_HIGH, "sw": DEFAULT_SWITCH, "gate": DEFAULT_GATE, "seed": DEFAULT_SEED},
    "Gleichmäßige Fahrt": {"sz": DEFAULT_SIGMA_Z, "out": 0.05, "dv": 0.0, "qh": DEFAULT_Q_HIGH, "sw": DEFAULT_SWITCH, "gate": DEFAULT_GATE, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard (Voreinstellung)": "5 % GPS-Ausreißer und ein Bremsmanöver von 6 m/s bei Sekunde 60: beides zugleich, wie im Stück Robuster Filter.",
    "Nur Manöver, keine Ausreißer": "Starkes Bremsmanöver (8 m/s) ohne Ausreißer: die Frage ist allein, wie schnell das Modell das Manöver annimmt.",
    "Viele Ausreißer": "Jede fünfte Messung ist ein Ausreißer, dazu ein Manöver.",
    "Gleichmäßige Fahrt": "Kein Manöver: was kostet das zweite Modell, wenn es nie gebraucht wird?",
}
