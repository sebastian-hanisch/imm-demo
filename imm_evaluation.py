"""Kennzahlen, Verdikte und Sweeps über die festen Instanzen (SWEEP_SEEDS)."""

from dataclasses import dataclass, replace

import numpy as np

import imm_algorithm as A
import imm_constants as C
import imm_scenario as S


@dataclass(frozen=True)
class Settings:
    sigma_z: float = C.DEFAULT_SIGMA_Z      # GPS-Rauschen (wahr und angenommen) in m
    out_share: float = C.DEFAULT_OUT        # Anteil der Ausreißer
    dv: float = C.DEFAULT_DV                # Geschwindigkeitsabnahme beim Manöver in m/s
    q_high: float = C.DEFAULT_Q_HIGH        # Beschleunigungsrauschen des unruhigen Modells
    p_switch: float = C.DEFAULT_SWITCH      # Modellwechsel-Wahrscheinlichkeit je Sekunde
    gate: float = C.DEFAULT_GATE            # Tor (nur beim IMM mit Tor)
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    track: S.Track
    runs: dict                 # Methode -> A.Run
    err: dict                  # Methode -> (T,) Positionsfehler mit Vorzeichen
    rmse: dict                 # Methode -> RMSE ab N_BURN
    lost: dict                 # Methode -> Endfehler über FAIL_ERR
    detect_delay: object       # Sekunden vom Manöverbeginn bis das unruhige Modell > 0,5 hat (None: nie / kein Manöver)
    detect_delay_gate: object


def _detect(mu_high, start):
    after = np.where(mu_high[start:] > C.MODE_THRESHOLD)[0]
    return int(after[0]) if len(after) else None


def run_all(settings, track):
    z = track.z
    s = settings
    r = {
        "kf_low": A.run_kf(z, C.DT, C.Q_LOW, s.sigma_z, C.V_START_MEAN, C.V_PRIOR_STD),
        "kf_high": A.run_kf(z, C.DT, s.q_high, s.sigma_z, C.V_START_MEAN, C.V_PRIOR_STD),
        "huber": A.run_kf(z, C.DT, C.Q_LOW, s.sigma_z, C.V_START_MEAN, C.V_PRIOR_STD, huber_c=C.HUBER_C),
        "imm": A.run_imm(z, C.DT, (C.Q_LOW, s.q_high), s.p_switch, s.sigma_z, C.V_START_MEAN, C.V_PRIOR_STD),
        "imm_gate": A.run_imm(z, C.DT, (C.Q_LOW, s.q_high), s.p_switch, s.sigma_z, C.V_START_MEAN, C.V_PRIOR_STD, gate_prob=s.gate),
    }
    return r


def analyse(settings):
    tr = S.simulate(settings.seed, settings.sigma_z, C.Q_LOW, settings.out_share, settings.dv)
    runs = run_all(settings, tr)
    sl = slice(C.N_BURN, None)
    err = {k: run.x[:, 0] - tr.s for k, run in runs.items()}
    rmse = {k: float(np.sqrt(np.mean(e[sl] ** 2))) for k, e in err.items()}
    lost = {k: bool(abs(e[-1]) > C.FAIL_ERR) for k, e in err.items()}
    has_maneuver = settings.dv > 0
    d1 = _detect(runs["imm"].mu[:, 1], C.MANEUVER_START) if has_maneuver else None
    d2 = _detect(runs["imm_gate"].mu[:, 1], C.MANEUVER_START) if has_maneuver else None
    return Analysis(settings, tr, runs, err, rmse, lost, d1, d2)


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def _over_seeds(settings):
    return [analyse(replace(settings, seed=s)) for s in C.SWEEP_SEEDS]


def _summary(results):
    return {k: {"rmse": float(np.median([a.rmse[k] for a in results])), "lost": float(np.mean([a.lost[k] for a in results]))} for k in C.METHODS}


def sweep_maneuver(settings):
    """Manöverstärke variieren: ohne Ausreißer und mit dem Ausreißeranteil der Seitenleiste."""
    rows = []
    for dv in C.DV_SWEEP:
        rows.append({"dv": dv, "clean": _summary(_over_seeds(replace(settings, dv=dv, out_share=0.0))),
                     "dirty": _summary(_over_seeds(replace(settings, dv=dv)))})
    return rows


def sweep_mode_probability(settings):
    """Mittlere Wahrscheinlichkeit des unruhigen Modells über die Zeit, einmal OHNE und einmal MIT Ausreißern (Ausreißeranteil der Seitenleiste);
    Manöver wie in der Seitenleiste. Dazu, wie oft und nach wie vielen Sekunden das Manöver ohne Ausreißer erkannt wird."""
    out = {}
    for name, share in (("clean", 0.0), ("dirty", settings.out_share)):
        res = _over_seeds(replace(settings, out_share=share))
        out[name] = {
            "mu": np.mean([a.runs["imm"].mu[:, 1] for a in res], axis=0),
            "mu_gate": np.mean([a.runs["imm_gate"].mu[:, 1] for a in res], axis=0),
        }
        delays = [d for d in (a.detect_delay for a in res) if d is not None]
        out[name]["detected"] = len(delays) / len(res)
        out[name]["median_delay"] = float(np.median(delays)) if delays else None
    out["t"] = np.arange(C.T_STEPS) * C.DT
    return out


def sweep_switch(settings):
    """Wechselwahrscheinlichkeit variieren (Manöver und Ausreißer wie in der Seitenleiste)."""
    rows = []
    for p in C.SWITCH_SWEEP:
        res = _over_seeds(replace(settings, p_switch=p))
        s = _summary(res)
        rows.append({"p": p, "imm": s["imm"], "imm_gate": s["imm_gate"], "kf_low": s["kf_low"], "huber": s["huber"]})
    return rows
