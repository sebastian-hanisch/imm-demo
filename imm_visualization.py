"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import imm_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _maneuver_rect(fig, dv):
    if dv > 0:
        fig.add_vrect(x0=C.MANEUVER_START, x1=C.MANEUVER_START + C.MANEUVER_STEPS, fillcolor="rgba(120,120,120,0.15)", line_width=0,
                      annotation_text="Manöver", annotation_position="top left")


def _style(k):
    return dict(color=C.COLORS[k], width=3 if k in ("imm", "imm_gate") else 2, dash="dot" if k == "kf_low" else "solid")


def build_error_chart(a):
    """Positionsfehler der fünf Verfahren über die Zeit."""
    t = a.track.t
    fig = go.Figure()
    for k in C.METHODS:
        fig.add_trace(go.Scatter(x=t, y=a.err[k], mode="lines", line=_style(k), name=C.METHOD_LABELS[k],
                                 hovertemplate="t = %{x:.0f} s: %{y:.1f} m<extra>" + C.METHOD_LABELS[k] + "</extra>"))
    _maneuver_rect(fig, a.settings.dv)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Positionsfehler (m)")
    return _base(fig)


def build_mode_chart(a):
    """Wahrscheinlichkeit des unruhigen Modells: IMM ohne und mit Tor."""
    t = a.track.t
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.runs["imm"].mu[:, 1], mode="lines", line=dict(color=C.COLORS["imm"], width=2), name=C.METHOD_LABELS["imm"]))
    fig.add_trace(go.Scatter(x=t, y=a.runs["imm_gate"].mu[:, 1], mode="lines", line=dict(color=C.COLORS["imm_gate"], width=3), name=C.METHOD_LABELS["imm_gate"]))
    rej = a.runs["imm_gate"].rejected
    fig.add_trace(go.Scatter(x=t[rej], y=np.full(rej.sum(), 1.02), mode="markers", marker=dict(color="#333", size=7, symbol="x"), name="vom Tor verworfen"))
    _maneuver_rect(fig, a.settings.dv)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Wahrscheinlichkeit des unruhigen Modells", range=[0, 1.08], tickformat=".0%")
    return _base(fig, 300)


def build_gps_chart(a):
    """GPS-Fehler jeder Messung: Ausreißer rot (zur Einordnung der Modellwahrscheinlichkeiten)."""
    t = a.track.t
    e = a.track.z - a.track.s
    out = a.track.outlier
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[~out], y=e[~out], mode="markers", marker=dict(color="rgba(150,150,150,0.8)", size=6), name="echte Messung"))
    fig.add_trace(go.Scatter(x=t[out], y=e[out], mode="markers", marker=dict(color="#b04a4a", size=9, symbol="diamond"), name="Ausreißer"))
    _maneuver_rect(fig, a.settings.dv)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="GPS-Fehler (m)")
    return _base(fig, 260)


def _lines(fig, xs, rows, key, field="rmse", methods=C.METHODS):
    for k in methods:
        fig.add_trace(go.Scatter(x=xs, y=[r[key][k][field] for r in rows], mode="lines+markers", line=_style(k), name=C.METHOD_LABELS[k]))


def build_maneuver_sweep(rows, dirty):
    fig = go.Figure()
    _lines(fig, [r["dv"] for r in rows], rows, "dirty" if dirty else "clean")
    fig.update_xaxes(title_text="Geschwindigkeitsabnahme beim Manöver (m/s)")
    fig.update_yaxes(title_text="Positionsfehler, RMSE (m)")
    return _base(fig)


def build_mode_sweep(res):
    t = res["t"]
    fig = go.Figure()
    for name, label, dash in (("clean", "ohne Ausreißer", "solid"), ("dirty", "mit Ausreißern", "dot")):
        fig.add_trace(go.Scatter(x=t, y=res[name]["mu_gate"], mode="lines", line=dict(color=C.COLORS["imm_gate"], width=3, dash=dash), name=f"IMM mit Tor, {label}"))
        fig.add_trace(go.Scatter(x=t, y=res[name]["mu"], mode="lines", line=dict(color=C.COLORS["imm"], width=2, dash=dash), name=f"IMM ohne Tor, {label}"))
    fig.add_vrect(x0=C.MANEUVER_START, x1=C.MANEUVER_START + C.MANEUVER_STEPS, fillcolor="rgba(120,120,120,0.15)", line_width=0)
    fig.update_xaxes(title_text="Zeit (s)")
    fig.update_yaxes(title_text="Mittlere Wahrscheinlichkeit des unruhigen Modells", range=[0, 1.0], tickformat=".0%")
    return _base(fig, 360)


def build_switch_sweep(rows):
    xs = [str(r["p"]).replace(".", ",") for r in rows]
    fig = go.Figure()
    for k in ("kf_low", "huber", "imm", "imm_gate"):
        fig.add_trace(go.Scatter(x=xs, y=[r[k]["rmse"] for r in rows], mode="lines+markers", line=_style(k), name=C.METHOD_LABELS[k]))
    fig.update_xaxes(title_text="Wechselwahrscheinlichkeit je Sekunde", type="category")
    fig.update_yaxes(title_text="Positionsfehler, RMSE (m)")
    return _base(fig)
