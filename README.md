# Mehrere Bewegungsmodelle: Manöver erkennen, ohne Ausreißer zu glauben – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-imm-demo.streamlit.app/)** -->

Achtes Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [Robuster Filter](https://github.com/sebastian-hanisch/robust-filter-demo) fort und behebt dessen benennbare Schwäche: **ein echtes Manöver sieht in der ersten Sekunde aus wie ein Ausreißer**,
und jede Schutzregel kostet im Manöver Genauigkeit, weil das Filter **ein** Bewegungsmodell kennt. Das IMM (Interacting Multiple Model) rechnet zwei Modelle (ruhig, unruhig) parallel und gewichtet sie
nach der Messwahrscheinlichkeit. Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Thema an einem wachsenden Beispiel.

## Warum dieses Problem

Wer ein Fahrzeug über längere Zeit verfolgt, sieht Phasen gleichmäßiger Fahrt und Manöver. Ein Modell muss entweder ruhig sein (glatt, aber im Manöver träge) oder unruhig (folgt dem Manöver, aber
verrauscht). Die Kernfrage: **Gewinnt man mit zwei Modellen mehr, als sie kosten?**

## Modell

Dieselbe 1D-Fahrt wie im Stück Robuster Filter (nur GPS, Ausreißer mit 80 m Streuung, optionales Bremsmanöver von Sekunde 60 bis 65). Zwei Modelle mit konstanter Geschwindigkeit: ruhig (0,3 m/s², gleich der
wahren Fahrunruhe) und unruhig (wählbar, 3 m/s²). Modellwechsel als Markov-Kette. Verglichen werden fünf Verfahren: ruhiges Modell, unruhiges Modell, ruhiges Modell mit Huber-Gewichtung, IMM, IMM mit Tor
(Messung verworfen, wenn sie für beide Modelle außerhalb des 99,9-%-Bereichs liegt). Formeln im 📐-Expander, Kern in `imm_algorithm.py`.

## Methodik

Drei Experimente auf Knopfdruck, je über 40 feste Fahrten (`SWEEP_SEEDS`):

1. Manöverstärke von 0 bis 10 m/s, ohne und mit Ausreißern: Positionsfehler der fünf Verfahren.
2. Wahrscheinlichkeit des unruhigen Modells über die Zeit und Erkennungsverzögerung.
3. Wechselwahrscheinlichkeit von 0,005 bis 0,2.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Median des Positionsfehlers (RMSE) über 40 feste Fahrten, GPS 10 m, unruhiges Modell 3 m/s², Wechselwahrscheinlichkeit 0,05; `tests/test_claims.py` rechnet jede nach.

| Frage | Befund |
|---|---|
| Was bringt das zweite Modell im Manöver (ohne Ausreißer)? | Bei 10 m/s Abnahme: ruhiges Modell **8,5 m**, Huber **9,8 m**, IMM mit Tor **5,8 m**. Der Fehler des IMM bleibt von 0 bis 10 m/s fast gleich (**5,3** bis **5,8 m**). |
| Was kostet es ohne Manöver? | **5,3 m** gegen **4,5 m** beim ruhigen Modell (etwa 18 % mehr); das unruhige Modell allein **6,4 m**. Mit 5 % Ausreißern ist Huber ohne Manöver besser (**5,1 m** gegen **6,3 m**). |
| Mit Ausreißern und starkem Manöver (5 %, 10 m/s) | IMM mit Tor **6,8 m**, Huber **10,5 m**, ruhiges Modell **11,2 m**. |
| Und das IMM ohne Tor? | Mit Ausreißern bei jeder Manöverstärke etwa **13,7 m**, mehr als doppelt so viel wie mit Tor; in 2,5 % der Fahrten verloren. Ein Ausreißer wird als Manöver gedeutet. |
| Wie schnell wird das Manöver erkannt? | **Langsam und unscharf**: ohne Ausreißer steigt die mittlere Wahrscheinlichkeit des unruhigen Modells von etwa **31 %** vor dem Manöver auf höchstens **57 %**, und die 50-%-Marke wird im Median **8 s** nach Beginn überschritten (Manöverdauer 5 s). |
| Wie wichtig ist die Wechselwahrscheinlichkeit? | **Unkritisch**: IMM mit Tor zwischen **6,4** und **6,8 m** über 0,005 bis 0,2, IMM ohne Tor um **13,7 m**. |
| Wie nah ist das IMM an der exakten Lösung? | Gegen die exakte Mischung über alle Modellfolgen (10 Sekunden um den Manöverbeginn, 512 Folgen): Modellwahrscheinlichkeit höchstens **0,012**, Position höchstens **0,42 m** daneben. |

## Ehrliche Grenzen

- **Die Wahrheit ist das Modell.** Die Fahrt hat genau die Dynamik, die das ruhige Modell annimmt, und ein Manöver, das das unruhige Modell abdeckt; echte Fahrten sind unordentlicher.
- **Das IMM wählt nur aus dem, was man ihm gibt.** Zwei von Hand gesetzte Modelle; eine Kurve oder ein Halt, den keines beschreibt, bleibt unerklärt.
- **Die Erkennung ist nicht scharf.** Das IMM mischt statt zu entscheiden; Modellwahrscheinlichkeiten sind hier keine verlässlichen Manöver-Detektoren.
- **Ohne Manöver ist das IMM schlechter als ein gutes Einzelmodell.** Bei 5 % Ausreißern ohne Manöver ist Huber besser.
- **Kein Radsensor, keine Glättung.** Ein Radsensor würde das Manöver direkt zeigen (siehe Stück 1); ein Glätter könnte Ausreißer und Manöver mit Blick in die Zukunft trennen.
- **Kleine Unterschiede sind Zufall.** Bei 40 Fahrten sind Unterschiede unter etwa 0,5 m nicht belastbar.

## Verifikation

- **Kern:** Wechselmatrix, Mischen (Handrechnung: gemischte Anfänge 2 und 8, Kovarianz 17), Kombinieren, Tor, Wahrscheinlichkeiten summieren zu 1 (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_imm.py`, anderer Rechenweg):
  - Ein Modell und zwei gleiche Modelle ergeben den gewöhnlichen Kalman-Filter.
  - **Exakte Gauß-Mischung über alle Modellfolgen** (512 Filter) gegen Modellwahrscheinlichkeiten und Mittelwert des IMM.
  - Huber mit unendlicher Schwelle gleich dem Kalman-Filter.
  - Angezeigte Kennzahlen und Erkennungsverzögerung gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, jede Auswahloption, Experiment-Knöpfe, Abgleich der angezeigten Kennzahlen mit `analyse(Settings(...))`) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `imm_algorithm.py` | Kern: Kalman-Filter, Huber, IMM (Mischen, Filtern, Gewichten, Kombinieren, Tor) |
| `imm_scenario.py` | Fahrt, Ausreißer, Manöver (ein Generator, Regler schwellen denselben Zufall) |
| `imm_evaluation.py` | Kennzahlen, Sweeps, Erkennungsverzögerung |
| `imm_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `imm_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `imm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `imm_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Mehr als zwei Modelle**, gelernte Modelle, Modelle mit Kurve oder Stillstand.
- **Echte GPS-Spuren, Radsensor.**
- **Verfahren mit mehr Hypothesen** (MHT).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
