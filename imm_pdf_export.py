"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF


def _num(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def build_pdf(a):
    """Einseitiger Bericht: Einstellungen, Kennzahlen, Verdikt. Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Mehrere Bewegungsmodelle (IMM)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("GPS-Rauschen", f"{_num(s.sigma_z, 1)} m"),
        ("Ausreißeranteil", f"{_num(100 * s.out_share, 0)} %"),
        ("Manöver (Geschwindigkeitsabnahme)", f"{_num(s.dv, 1)} m/s"),
        ("Unruhiges Modell", f"{_num(s.q_high, 1)} m/s²"),
        ("Wechselwahrscheinlichkeit je Sekunde", _num(s.p_switch, 3)),
        ("Tor", f"{_num(100 * s.gate, 2)} %"),
        ("Zufalls-Seed", str(s.seed)),
    ]
    for k, v in rows:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Fehler ein Modell, ruhig (RMSE)", f"{_num(a.rmse['kf_low'])} m"),
        ("Fehler ein Modell, unruhig (RMSE)", f"{_num(a.rmse['kf_high'])} m"),
        ("Fehler Huber (RMSE)", f"{_num(a.rmse['huber'])} m"),
        ("Fehler IMM (RMSE)", f"{_num(a.rmse['imm'])} m"),
        ("Fehler IMM mit Tor (RMSE)", f"{_num(a.rmse['imm_gate'])} m"),
    ]
    for k, v in metrics:
        pdf.cell(95, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Synthetische Fahrt mit bekanntem Modell; die Zahlen gelten für dieses Szenario und diese Einstellungen.")
    return bytes(pdf.output())
