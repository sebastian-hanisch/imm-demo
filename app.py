"""Mehrere Bewegungsmodelle (IMM): Manöver erkennen, ohne Ausreißer zu glauben - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dieselbe Fahrt wie im Stück „Robuster Filter“ (GPS-Ausreißer, optionales Bremsmanöver). Hier läuft statt eines Bewegungsmodells ein Paar
(ruhig und unruhig), gewichtet nach der Messwahrscheinlichkeit: das IMM (Interacting Multiple Model).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import imm_constants as C
import imm_evaluation as ev
from imm_evaluation import Settings, analyse
from imm_pdf_export import build_pdf
from imm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from imm_visualization import build_error_chart, build_gps_chart, build_maneuver_sweep, build_mode_chart, build_mode_sweep, build_switch_sweep

st.set_page_config(page_title="Mehrere Bewegungsmodelle – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


def pct_label(p):
    return f"{100 * p:g} %".replace(".", ",")


def switch_label(p):
    return f"{p:g}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_maneuver(settings):
    return ev.sweep_maneuver(settings)


@st.cache_data(show_spinner=False)
def _sweep_mode(settings):
    return ev.sweep_mode_probability(settings)


@st.cache_data(show_spinner=False)
def _sweep_switch(settings):
    return ev.sweep_switch(settings)


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben."""
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (einige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner einige Sekunden)"):
        return fn(settings)


st.title("🔀 Mehrere Bewegungsmodelle: Manöver erkennen, ohne Ausreißer zu glauben")
st.markdown(
    """
Im Stück **Robuster Filter** blieb ein Dilemma: ein echtes Manöver sieht in der ersten Sekunde aus wie ein Ausreißer, und jede Schutzregel kostet im Manöver Genauigkeit. Der Grund liegt im **einen**
Bewegungsmodell, das nur „fast konstante Geschwindigkeit“ kennt. Das **IMM (Interacting Multiple Model)** rechnet **zwei** Modelle gleichzeitig: ein **ruhiges** (kleine zufällige Beschleunigung) und
ein **unruhiges** (große). Jede Sekunde wird nach der Messwahrscheinlichkeit gewichtet, **welches Modell gerade stimmt**. Bremst das Fahrzeug, gewinnt das unruhige Modell, und die Schätzung
folgt dem Manöver; fährt es gleichmäßig, gewinnt das ruhige. Die Demo ist das achte Stück der Linie **Zustandsschätzung und Filter**. Im Unterschied zu den Fall-Demos zeigt sie ein einziges
Thema an einem wachsenden Beispiel.

Weiter unten: drei Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das IMM", expanded=True):
    st.markdown(
        """
Pro Sekunde, für zwei Modelle (ruhig, unruhig):

1. **Mischen:** Jedes Modell startet nicht bei sich selbst, sondern bei einer **Mischung** beider letzter Schätzungen, gewichtet mit der Wahrscheinlichkeit, aus dem anderen Modell herübergewechselt
   zu sein (Wechselwahrscheinlichkeit je Sekunde).
2. **Filtern:** Beide Modelle machen Vorhersage und Korrektur, jedes mit seinem Beschleunigungsrauschen.
3. **Gewichten:** Je Modell: wie wahrscheinlich war die Messung unter diesem Modell? Daraus die neue **Modellwahrscheinlichkeit**.
4. **Ausgeben:** gewichteter Mittelwert beider Schätzungen.

**Das Tor** (hier optional): Liegt eine Messung für **beide** Modelle außerhalb des Bereichs, den 99,9 % der echten Messungen erreichen, wird sie als Ausreißer verworfen. Da das unruhige
Modell breit ist, zählt eine Abweichung, die es noch erklärt, nicht als Ausreißer: das Tor sperrt das Manöver nicht aus.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Messung**")
    sigma_z = st.slider("GPS-Rauschen (m)", *bounds("sigma_slider"), key="sigma_slider", step=0.5, help="Streuung der normalen GPS-Messungen. Die Filter kennen diesen Wert.")
    out = st.slider("Ausreißeranteil", *bounds("out_slider"), key="out_slider", step=0.01, format="%.2f",
                    help="Anteil der Messungen mit einem Fehler von 80 m Streuung statt des normalen Rauschens.")
    st.markdown("**Die Fahrt**")
    dv = st.slider("Bremsmanöver (m/s)", *bounds("dv_slider"), key="dv_slider", step=1.0,
                   help="Geschwindigkeitsabnahme ab Sekunde 60, verteilt auf 5 s. Null heißt: gleichmäßige Fahrt.")
    st.markdown("**Die Modelle**")
    q_high = st.slider("Unruhiges Modell (m/s²)", *bounds("qh_slider"), key="qh_slider", step=0.5,
                       help="Beschleunigungsrauschen des unruhigen Modells. Das ruhige Modell kennt die wahre Fahrunruhe (0,3 m/s²).")
    p_switch = st.select_slider("Wechselwahrscheinlichkeit je Sekunde", C.SWITCH_CHOICES, key="switch_select", format_func=switch_label,
                                help="Wie leicht das IMM zwischen den Modellen wechselt. Kleine Werte machen es träge, große nervös.")
    gate = st.select_slider("Tor lässt echte Messungen durch", C.GATE_CHOICES, key="gate_select", format_func=pct_label,
                            help="Nur für „Zwei Modelle + Tor“: Anteil echter Messungen, den das Tor durchlassen soll.")
    seed = st.number_input("Zufalls-Seed der Fahrt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrt generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"sigma_slider": float(sigma_z), "out_slider": float(out), "dv_slider": float(dv), "qh_slider": float(q_high),
                   "switch_select": float(p_switch), "gate_select": float(gate), "seed_input": int(seed)})

settings = Settings(sigma_z=float(sigma_z), out_share=float(out), dv=float(dv), q_high=float(q_high), p_switch=float(p_switch), gate=float(gate), seed=int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m = st.columns(5)
for col, k in zip(m, C.METHODS):
    suffix = "  ⚠ verloren" if a.lost[k] else ""
    delta = "Referenz" if k == "kf_low" else f"{a.rmse[k] - a.rmse['kf_low']:+.1f} m gegen ruhiges Modell".replace(".", ",")
    col.metric(C.METHOD_LABELS[k], f"{de(a.rmse[k])} m{suffix}", delta=delta, delta_color="off" if k == "kf_low" else "inverse",
               help="Wurzel des mittleren quadratischen Positionsfehlers (RMSE) ab Sekunde 10. „Verloren“: Endfehler über 50 m.")

st.markdown("**Positionsfehler über die Zeit**")
st.plotly_chart(build_error_chart(a), width="stretch", key="error_chart")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Welches Modell gerade stimmt** – Wahrscheinlichkeit des unruhigen Modells, Kreuze: vom Tor verworfene Messungen.")
    st.plotly_chart(build_mode_chart(a), width="stretch", key="mode_chart")
with c2:
    st.markdown("**Was der Sensor liefert** – GPS-Fehler je Messung, rot: Ausreißer.")
    st.plotly_chart(build_gps_chart(a), width="stretch", key="gps_chart")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="imm_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Gewinnt man mit zwei Modellen mehr, als sie kosten?")
r = a.rmse
best_single = min(r["kf_low"], r["kf_high"], r["huber"])
if r["imm_gate"] < 0.9 * best_single:
    st.success(
        f"**Ja, hier:** IMM mit Tor {de(r['imm_gate'])} m gegen {de(best_single)} m für das beste Einzelmodell-Verfahren (ruhig {de(r['kf_low'])}, unruhig {de(r['kf_high'])}, "
        f"Huber {de(r['huber'])} m)."
    )
elif r["imm_gate"] > 1.1 * best_single:
    st.info(
        f"**Hier nicht:** IMM mit Tor {de(r['imm_gate'])} m gegen {de(best_single)} m des besten Einzelmodell-Verfahrens. Ohne Manöver oder mit wenig Manöver zahlt man für das zweite Modell "
        "und gewinnt nichts."
    )
else:
    st.info(f"**Gleichauf:** IMM mit Tor {de(r['imm_gate'])} m, bestes Einzelmodell-Verfahren {de(best_single)} m.")
if a.lost["imm"] and not a.lost["imm_gate"]:
    st.warning("Das IMM **ohne Tor** hat die Spur verloren (ein Ausreißer wird als Manöver gedeutet); mit Tor nicht.")
st.caption("Eine einzelne Fahrt zeigt nur, was hier passiert ist; die Experimente unten mitteln über 40 feste Fahrten.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie hängt alles von der Manöverstärke ab?")
st.caption(f"Median über {len(C.SWEEP_SEEDS)} feste Fahrten; links ohne Ausreißer, rechts mit dem Ausreißeranteil der Seitenleiste; übrige Regler wie dort.")
rows = run_experiment("maneuver", _sweep_maneuver, settings, f"▶ Experiment 1 rechnen ({2 * len(C.DV_SWEEP) * len(C.SWEEP_SEEDS)} Fahrten)")
if rows is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ohne Ausreißer**")
        st.plotly_chart(build_maneuver_sweep(rows, dirty=False), width="stretch", key="maneuver_clean")
    with c2:
        st.markdown("**Mit Ausreißern**")
        st.plotly_chart(build_maneuver_sweep(rows, dirty=True), width="stretch", key="maneuver_dirty")
    r0 = rows[0]
    r10 = rows[-1]
    st.info(
        f"Ohne Manöver kostet das zweite Modell Genauigkeit: {de(r0['clean']['imm_gate']['rmse'])} m gegen {de(r0['clean']['kf_low']['rmse'])} m beim ruhigen Modell. "
        f"Bei einer Abnahme von {r10['dv']:.0f} m/s kehrt es sich um: IMM mit Tor {de(r10['clean']['imm_gate']['rmse'])} m, ruhiges Modell {de(r10['clean']['kf_low']['rmse'])} m, Huber "
        f"{de(r10['clean']['huber']['rmse'])} m. **Der Fehler des IMM bleibt über alle Manöverstärken fast gleich.** Mit Ausreißern ist das IMM ohne Tor "
        f"({de(r10['dirty']['imm']['rmse'])} m) schlechter als jedes andere Verfahren, mit Tor {de(r10['dirty']['imm_gate']['rmse'])} m."
    )

st.subheader("🔬 Experiment 2: Wie schnell erkennt das IMM das Manöver?")
st.caption(f"Mittel über {len(C.SWEEP_SEEDS)} feste Fahrten; Manöver wie in der Seitenleiste, einmal ohne und einmal mit dem Ausreißeranteil der Seitenleiste. Grau hinterlegt: das Manöver.")
mres = run_experiment("mode", _sweep_mode, settings, f"▶ Experiment 2 rechnen ({2 * len(C.SWEEP_SEEDS)} Fahrten)")
if mres is not None:
    st.plotly_chart(build_mode_sweep(mres), width="stretch", key="mode_sweep")
    cl = mres["clean"]
    pre_c = float(cl["mu_gate"][20:C.MANEUVER_START - 2].mean())
    post_c = float(cl["mu_gate"][C.MANEUVER_START:C.MANEUVER_START + 25].max())
    pre_d = float(mres["dirty"]["mu"][20:C.MANEUVER_START - 2].mean())
    delay = cl["median_delay"]
    st.info(
        f"Ohne Ausreißer steigt die Wahrscheinlichkeit des unruhigen Modells von im Mittel {pct(pre_c)} vor dem Manöver auf höchstens {pct(post_c)} danach"
        + (f"; die 50-%-Marke überschreitet sie im Median {delay:.0f} s nach Manöverbeginn (Manöverdauer {C.MANEUVER_STEPS} s)" if delay is not None and settings.dv > 0 else "")
        + f". **Die Erkennung ist langsam und unscharf**: das IMM wartet nicht auf Gewissheit, es mischt. Mit Ausreißern hebt das IMM ohne Tor die Wahrscheinlichkeit schon vor dem Manöver auf {pct(pre_d)}, "
        "weil jeder Ausreißer das unruhige Modell stützt; das Tor verhindert das."
    )

st.subheader("🔬 Experiment 3: Wie träge darf das IMM sein?")
st.caption(f"Median über {len(C.SWEEP_SEEDS)} feste Fahrten; Manöver, Ausreißer und übrige Regler wie in der Seitenleiste, die Wechselwahrscheinlichkeit läuft von 0,005 bis 0,2.")
srows = run_experiment("switch", _sweep_switch, settings, f"▶ Experiment 3 rechnen ({len(C.SWITCH_SWEEP) * len(C.SWEEP_SEEDS)} Fahrten)")
if srows is not None:
    st.plotly_chart(build_switch_sweep(srows), width="stretch", key="switch_sweep")
    lo = min(r["imm_gate"]["rmse"] for r in srows)
    hi = max(r["imm_gate"]["rmse"] for r in srows)
    st.info(
        f"Das IMM mit Tor liegt über alle Wechselwahrscheinlichkeiten zwischen {de(lo)} und {de(hi)} m, das IMM ohne Tor bleibt bei etwa {de(srows[0]['imm']['rmse'])} m: die Einstellung "
        "ist hier **unkritisch**. Wichtiger ist, **dass** es ein Tor gibt und wie breit das unruhige Modell ist."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Modellmenge deckt die Wirklichkeit ab** | Das IMM wählt nur zwischen den Modellen, die man ihm gibt. Eine Kurvenfahrt, ein Anhalten oder ein Sprung, den keines beschreibt, bleibt unerklärt. | Weitere Modelle (Kurve, Stillstand), gelernte Modelle |
| **Die Modelle sind gut abgestimmt** | Das unruhige Modell ist hier von Hand gesetzt (3 m/s²). Ein zu enges Modell erklärt das Manöver nicht, ein zu weites glättet zu wenig. | Parameter aus Daten schätzen |
| **Zwei Modelle genügen** | Mehr Modelle erlauben feinere Abstufungen, kosten aber Rechenzeit und machen die Modellwahrscheinlichkeiten unschärfer. | – |
| **Die Mischung ist eine Näherung** | Das IMM fasst die Hypothesenfolgen je Sekunde zu je einem Gauß-Term pro Modell zusammen; exakt wäre eine Mischung über alle Modellfolgen. | Verfahren mit mehr Hypothesen (MHT) |
| **Ausreißer sind selten und einzeln** | Serien von Ausreißern können wie ein Manöver aussehen; das Tor hilft dann nur begrenzt. | Mehrsekunden-Entscheidungen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modelle.** $m = 2$ Modelle $M_j$: $x_{k+1} = F x_k + w$, $w \sim \mathcal N(0, Q_j)$ mit $Q_j = q_j^2\begin{pmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{pmatrix}$,
$q_1 = 0{,}3$ m/s² (ruhig), $q_2$ wählbar (unruhig). Das aktive Modell wechselt als Markov-Kette mit Übergangsmatrix $p_{ij}$ (hier $1 - p$ Bleiben, $p$ Wechseln).

**Ein Schritt.** Mit Modellwahrscheinlichkeiten $\mu_i(k-1)$:
1. Mischwahrscheinlichkeiten $\bar c_j = \sum_i p_{ij}\mu_i$ und $\mu_{i\mid j} = p_{ij}\mu_i/\bar c_j$; gemischter Anfangszustand $\hat x^0_j = \sum_i \mu_{i\mid j}\hat x_i$, Kovarianz
   $P^0_j = \sum_i \mu_{i\mid j}\bigl[P_i + (\hat x_i - \hat x^0_j)(\hat x_i - \hat x^0_j)^\top\bigr]$.
2. Je Modell Vorhersage und Korrektur ab $(\hat x^0_j, P^0_j)$ mit $Q_j$; Likelihood $\Lambda_j = \mathcal N(\nu_j; 0, S_j)$.
3. Moduswahrscheinlichkeit $\mu_j = \Lambda_j\bar c_j / \sum_l \Lambda_l\bar c_l$.
4. Ausgabe $\hat x = \sum_j \mu_j \hat x_j$, $P = \sum_j \mu_j\bigl[P_j + (\hat x_j - \hat x)(\hat x_j - \hat x)^\top\bigr]$.

**Tor.** Die Messung wird nur eingearbeitet, wenn für **mindestens ein** Modell $\mathrm{NIS}_j = \nu_j^2/S_j \le \gamma$ gilt; sonst bleibt $\mu = \bar c$ und die Korrektur entfällt.

**Näherung.** Exakt wäre eine Gauß-Mischung über alle $2^k$ Modellfolgen; das IMM fasst nach jedem Schritt je Zielmodell zusammen (Merging).

**Literatur.** H. A. P. Blom, Y. Bar-Shalom (1988), *The interacting multiple model algorithm for systems with Markovian switching coefficients*, IEEE Transactions on Automatic Control 33(8), 780–783
(Herkunft des Verfahrens; die Schrittfolge oben ist die übliche Darstellung).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
