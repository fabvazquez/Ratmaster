"""
ui/dialogs/radiobio.py  —  Análisis radiobiológico TCP / NTCP / BED / EQD2
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from PySide6 import QtCore, QtWidgets
from ratmaster.ui.canvas import MplCanvas
from ratmaster.physics.radiobio.models import (
    compute_radiobio_report, RadiobioOrganParams, TISSUE_DEFAULTS,
)
from ratmaster.physics.radiobio.tcp  import tcp_dose_curve
from ratmaster.physics.radiobio.ntcp import ntcp_dose_curve
from ratmaster.physics.radiobio.eud  import geud

_COLORS = ["#E53935","#1E88E5","#43A047","#8E24AA",
           "#FB8C00","#00ACC1","#6D4C41","#F06292","#AED581","#546E7A"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cell(txt, align=QtCore.Qt.AlignCenter):
    item = QtWidgets.QTableWidgetItem(str(txt))
    item.setFlags(QtCore.Qt.ItemIsEnabled)
    item.setTextAlignment(int(align))
    return item


def _info(html: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(html)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("background:#eef2ff; padding:6px; border-radius:4px; "
                      "font-size:12px;")
    return lbl


def _find_scale_at(A_arr: np.ndarray, aR, bR, GR, N0, target: float,
                   hi=30.0) -> float | None:
    """Escala de dosis donde TCP = target (fracción). None si no existe en [0, hi]."""
    def obj(s):
        S = np.exp(-aR * A_arr * s - GR * bR * (A_arr * s)**2)
        return float(np.exp(-N0 * float(np.mean(S)))) - target
    try:
        if obj(1e-6) * obj(hi) > 0:
            return None
        return float(brentq(obj, 1e-6, hi, xtol=1e-3))
    except Exception:
        return None


def _find_ntcp_scale(A_arr, TD50, m, n, model, target, hi=30.0) -> float | None:
    from ratmaster.physics.radiobio.ntcp import ntcp_lkb, ntcp_logistic
    fn = ntcp_lkb if model == "lkb" else ntcp_logistic
    def obj(s):
        return fn(A_arr * s, TD50, m, n) - target
    try:
        if obj(1e-6) * obj(hi) > 0:
            return None
        return float(brentq(obj, 1e-6, hi, xtol=1e-3))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Diálogo principal
# ─────────────────────────────────────────────────────────────────────────────

class RadiobioDialog(QtWidgets.QDialog):

    def __init__(self, parent, report: dict, isoe_params_by_organ: dict):
        super().__init__(parent)
        self.setWindowTitle("Análisis Radiobiológico — TCP / NTCP / BED")
        self.resize(980, 440)

        self.report               = report
        self.isoe_params_by_organ = isoe_params_by_organ
        self.radiobio_report: dict | None = None
        self.organs = sorted(report.get("IsoVoxel", {}).keys())

        self.rb_params: dict[str, RadiobioOrganParams] = {}
        self._init_defaults()
        self._tbl_w: dict[str, dict] = {}

        self._build_ui()

    # ── defaults ─────────────────────────────────────────────────────────────
    def _init_defaults(self):
        for organ in self.organs:
            k = organ.lower()
            if any(h in k for h in ["tumor","gtv","ctv","melanoma","glioma",
                                      "neoplasia","carcinoma","cancer","gbm"]):
                p = RadiobioOrganParams(organ_type="tumor", N0=1e7, alpha_beta=10.0)
            elif any(h in k for h in ["cerebro","brain","encefalo"]):
                p = RadiobioOrganParams(**vars(
                    TISSUE_DEFAULTS.get("normal_brain", RadiobioOrganParams())))
            elif any(h in k for h in ["spinal","medula","cord"]):
                p = RadiobioOrganParams(**vars(
                    TISSUE_DEFAULTS.get("spinal_cord", RadiobioOrganParams())))
            elif any(h in k for h in ["skin","piel"]):
                p = RadiobioOrganParams(**vars(
                    TISSUE_DEFAULTS.get("skin", RadiobioOrganParams())))
            else:
                p = RadiobioOrganParams(organ_type="oar")
            self.rb_params[organ] = p

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setContentsMargins(6, 6, 6, 4)
        inner_layout.setSpacing(4)

        self.tabs = QtWidgets.QTabWidget()
        inner_layout.addWidget(self.tabs)

        self.tabs.addTab(self._tab_config(),    "Configuración")
        self.tabs.addTab(self._tab_results(),   "TCP / NTCP")
        self.tabs.addTab(self._tab_bed(),       "BED / EQD2")

        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)
        brow = QtWidgets.QHBoxLayout()
        brow.setContentsMargins(6, 0, 6, 0)
        self.btn_calc = QtWidgets.QPushButton("▶  Calcular")
        self.btn_calc.setDefault(True)
        btn_close = QtWidgets.QPushButton("Cerrar")
        brow.addWidget(self.btn_calc); brow.addStretch(); brow.addWidget(btn_close)
        root.addLayout(brow)

        self.btn_calc.clicked.connect(self._run)
        btn_close.clicked.connect(self.accept)

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    def _tab_config(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4); v.setSpacing(4)

        v.addWidget(_info(
            "<b>Parámetros por órgano.</b> "
            "Elegí <i>tumor</i> para calcular TCP (control tumoral) o "
            "<i>oar</i> para NTCP (complicación en tejido sano).<br>"
            "• <b>N₀</b>: cuántas células tiene el tumor (tumor). "
            "• <b>TD₅₀</b>: dosis que daña el 50% de los pacientes (oar). "
            "• <b>m</b>: qué tan pronunciada es la curva de complicación (0.1–0.2). "
            "• <b>n</b>: importancia del volumen irradiado (≈0.05 médula, ≈0.87 pulmón). "
            "• <b>α/β</b>: sensibilidad al fraccionamiento (alto=tumor, bajo=tejido tardío)."
        ))

        cols = ["Órgano","Tipo","N₀ (céls.)","TD₅₀ (Gy)","m","n","α/β (Gy)","Modelo NTCP"]
        tbl = QtWidgets.QTableWidget(len(self.organs), len(cols))
        tbl.setHorizontalHeaderLabels(cols)
        tbl.verticalHeader().setVisible(False)
        tbl.setAlternatingRowColors(True)
        tbl.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        tbl.horizontalHeader().setStretchLastSection(True)
        self.config_tbl = tbl

        for row, organ in enumerate(self.organs):
            p = self.rb_params[organ]
            rw: dict = {}

            item = QtWidgets.QTableWidgetItem(organ)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            tbl.setItem(row, 0, item)

            cb_type = QtWidgets.QComboBox()
            cb_type.addItems(["oar","tumor"])
            cb_type.setCurrentText(p.organ_type)
            tbl.setCellWidget(row, 1, cb_type); rw["type"] = cb_type

            sp_N0 = QtWidgets.QDoubleSpinBox()
            sp_N0.setRange(1e3, 1e12); sp_N0.setDecimals(0); sp_N0.setSingleStep(1e6)
            sp_N0.setValue(p.N0); sp_N0.setEnabled(p.organ_type == "tumor")
            tbl.setCellWidget(row, 2, sp_N0); rw["N0"] = sp_N0

            sp_td = QtWidgets.QDoubleSpinBox()
            sp_td.setRange(1.0, 300.0); sp_td.setDecimals(1); sp_td.setValue(p.TD50)
            sp_td.setEnabled(p.organ_type == "oar")
            tbl.setCellWidget(row, 3, sp_td); rw["TD50"] = sp_td

            sp_m = QtWidgets.QDoubleSpinBox()
            sp_m.setRange(0.01, 1.0); sp_m.setDecimals(3); sp_m.setValue(p.m)
            sp_m.setEnabled(p.organ_type == "oar")
            tbl.setCellWidget(row, 4, sp_m); rw["m"] = sp_m

            sp_n = QtWidgets.QDoubleSpinBox()
            sp_n.setRange(0.001, 1.5); sp_n.setDecimals(3); sp_n.setValue(p.n)
            sp_n.setEnabled(p.organ_type == "oar")
            tbl.setCellWidget(row, 5, sp_n); rw["n"] = sp_n

            sp_ab = QtWidgets.QDoubleSpinBox()
            sp_ab.setRange(0.1, 50.0); sp_ab.setDecimals(1); sp_ab.setValue(p.alpha_beta)
            tbl.setCellWidget(row, 6, sp_ab); rw["ab"] = sp_ab

            cb_mod = QtWidgets.QComboBox()
            cb_mod.addItems(["lkb","logistic"])
            cb_mod.setCurrentText(p.ntcp_model)
            cb_mod.setEnabled(p.organ_type == "oar")
            tbl.setCellWidget(row, 7, cb_mod); rw["ntcp_model"] = cb_mod

            def _sync(text, r=rw):
                is_t = (text == "tumor")
                r["N0"].setEnabled(is_t); r["TD50"].setEnabled(not is_t)
                r["m"].setEnabled(not is_t); r["n"].setEnabled(not is_t)
                r["ntcp_model"].setEnabled(not is_t)
            cb_type.currentTextChanged.connect(_sync)
            self._tbl_w[organ] = rw

        v.addWidget(tbl, stretch=1)
        return w

    # ── Tab 2 ─────────────────────────────────────────────────────────────────
    def _tab_results(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4); v.setSpacing(4)

        v.addWidget(_info(
            "<b>TCP</b> (control tumoral): probabilidad de que NINGUNA célula tumoral sobreviva. "
            "Querés TCP → 100%.<br>"
            "<b>NTCP</b> (complicación): probabilidad de daño clínico grave en tejido sano. "
            "Querés NTCP → 0%.<br>"
            "<b>gEUD</b>: dosis única equivalente con el mismo efecto biológico que la distribución real.<br>"
            "<b>Factor D₅₀</b>: cuántas veces habría que multiplicar la dosis actual para llegar al 50% de TCP o NTCP. "
            "Si es 1.4 → la dosis actual es el 71% de lo necesario para el 50% de efecto.<br>"
            "<b>IC 90% (MC)</b>: intervalo de confianza calculado propagando la incertidumbre "
            "de la dosis vóxel a vóxel. Solo aparece si el reporte tiene SigmaIsoVoxel."
        ))

        self.lbl_summary = QtWidgets.QLabel("")
        self.lbl_summary.setStyleSheet("font-weight:bold; padding:4px; font-size:13px;")
        v.addWidget(self.lbl_summary)

        cols = ["Órgano","Tipo","TCP / NTCP","± σ (MC)","IC 90% (MC)",
                "gEUD (Gy)","D_media (Gy)","D_max (Gy)","Factor D₅₀"]
        self.tbl_r = QtWidgets.QTableWidget(0, len(cols))
        self.tbl_r.setHorizontalHeaderLabels(cols)
        self.tbl_r.verticalHeader().setVisible(False)
        self.tbl_r.setAlternatingRowColors(True)
        self.tbl_r.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_r.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_r, stretch=1)

        v.addWidget(_info(
            "📈 <b>Curvas:</b> TCP/NTCP en función del factor de escala de la dosis. "
            "El eje X se ajusta automáticamente para mostrar la zona sigmoidal de cada curva. "
            "La línea vertical gris = dosis actual (escala 1)."
        ))
        self.canvas_rbc = MplCanvas(w, width=7, height=2.5)
        v.addWidget(self.canvas_rbc)
        return w

    # ── Tab 4 ─────────────────────────────────────────────────────────────────
    def _tab_bed(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4); v.setSpacing(4)

        v.addWidget(_info(
            "<b>BED</b> (Dosis Biológicamente Efectiva): mide el daño biológico total, "
            "independientemente de si se irradia en una o muchas sesiones. "
            "Fórmula: BED = A × (1 + A / (α/β)).<br>"
            "<b>EQD2</b> (Equivalente en fracciones de 2 Gy): convierte la dosis BNCT "
            "a cuánto sería si se hubiera administrado en sesiones de 2 Gy. "
            "Permite comparar directamente con límites de la literatura de radioterapia: "
            "p.ej. cerebro ≤ 60 Gy, médula espinal ≤ 45–50 Gy, piel ≤ 55 Gy.<br>"
            "Todas las columnas en Gy fotónico equivalente. D₅₀ = mediana de la distribución."
        ))

        cols = ["Órgano","Tipo","α/β (Gy)",
                "BED media","BED D₅₀","BED máx",
                "EQD2 media","EQD2 D₅₀","EQD2 máx"]
        self.tbl_b = QtWidgets.QTableWidget(0, len(cols))
        self.tbl_b.setHorizontalHeaderLabels(cols)
        self.tbl_b.verticalHeader().setVisible(False)
        self.tbl_b.setAlternatingRowColors(True)
        self.tbl_b.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_b.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_b, stretch=1)
        return w

    # ── collect ───────────────────────────────────────────────────────────────
    def _collect(self) -> dict[str, RadiobioOrganParams]:
        out = {}
        for organ, rw in self._tbl_w.items():
            out[organ] = RadiobioOrganParams(
                organ_type = rw["type"].currentText(),
                N0         = float(rw["N0"].value()),
                TD50       = float(rw["TD50"].value()),
                m          = float(rw["m"].value()),
                n          = float(rw["n"].value()),
                alpha_beta = float(rw["ab"].value()),
                ntcp_model = rw["ntcp_model"].currentText(),
                run_mc     = True,
            )
        return out

    # ── run ───────────────────────────────────────────────────────────────────
    def _run(self):
        self.btn_calc.setEnabled(False)
        self.btn_calc.setText("⏳ Calculando…")
        QtWidgets.QApplication.processEvents()
        try:
            self.radiobio_report = compute_radiobio_report(
                isoe_report          = self.report,
                params_by_organ      = self._collect(),
                isoe_params_by_organ = self.isoe_params_by_organ,
                N_mc_samples         = 500,
                run_mc               = True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        finally:
            self.btn_calc.setEnabled(True)
            self.btn_calc.setText("▶  Calcular")

        self._upd_results()
        self._upd_bed()
        self.tabs.setCurrentIndex(1)

    # ── update Tab 2 ─────────────────────────────────────────────────────────
    def _upd_results(self):
        rep    = self.radiobio_report
        summ   = rep.get("summary", {})
        params = self._collect()
        iso_vox = self.report.get("IsoVoxel", {})

        # Resumen en lenguaje simple
        tcp_t = summ.get("TCP_total")
        max_n = summ.get("max_NTCP")
        parts = []
        if tcp_t is not None:
            interp = ("excelente ✔" if tcp_t > 0.9 else
                      "buena" if tcp_t > 0.5 else
                      "insuficiente con estos parámetros ✗" if tcp_t < 0.01 else "parcial")
            parts.append(f"TCP total: <b>{tcp_t:.1%}</b> ({interp})")
        if max_n is not None:
            interp = ("mínima ✔" if max_n < 0.05 else
                      "moderada" if max_n < 0.20 else "alta ✗")
            parts.append(f"NTCP máximo: <b>{max_n:.1%}</b> ({interp})")
        self.lbl_summary.setText("   ".join(parts))

        organs_data = rep.get("organs", {})
        tbl = self.tbl_r
        tbl.setRowCount(len(organs_data))


        ax = self.canvas_rbc.ax
        ax.clear()

        for r, (organ, data) in enumerate(organs_data.items()):
            otype  = data.get("organ_type", "oar")
            iso_p  = self.isoe_params_by_organ.get(organ)
            p_rb   = params.get(organ)
            A      = np.asarray(iso_vox.get(organ, []), float)
            color  = _COLORS[r % len(_COLORS)]
            mc     = data.get("mc", {})

            tbl.setItem(r, 0, _cell(organ, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter))
            tbl.setItem(r, 1, _cell(otype))

            if A.size > 0 and p_rb:
                a_par = 1.0 / max(p_rb.n, 1e-6) if otype == "oar" else 1.0
                tbl.setItem(r, 5, _cell(f"{geud(A, a_par):.2f}"))
            else:
                tbl.setItem(r, 5, _cell("—"))
            tbl.setItem(r, 6, _cell(f"{np.mean(A):.2f}" if A.size > 0 else "—"))
            tbl.setItem(r, 7, _cell(f"{np.max(A):.2f}"  if A.size > 0 else "—"))

            if otype == "tumor" and "tcp" in data and iso_p and p_rb and A.size > 0:
                val    = data["tcp"]["TCP"]
                D_mean = float(np.mean(A))
                tbl.setItem(r, 2, _cell(f"TCP = {val:.2%}"))
                tbl.setItem(r, 3, _cell(f"±{mc.get('TCP_std',0):.2%}" if mc else "sin σ(A)"))
                tbl.setItem(r, 4, _cell(
                    f"[{mc.get('TCP_p5',val):.2%} – {mc.get('TCP_p95',val):.2%}]"
                    if mc else "sin σ(A)"))

                # Curva teórica TCP vs. dosis uniforme [Gy]
                # Rango: 0 hasta 3× D_mean o hasta donde TCP > 99%
                d_max  = max(D_mean * 3.5, 30.0)
                dose_x = np.linspace(0.0, d_max, 400)
                curve  = tcp_dose_curve(dose_x, iso_p.aR, iso_p.bR, iso_p.GR, p_rb.N0)
                ax.plot(dose_x, curve * 100, color=color, lw=2,
                        linestyle="--", label=f"TCP {organ}")
                # Punto del plan actual: (D_mean, TCP_real)
                ax.plot(D_mean, val * 100, "o", color=color, ms=8, zorder=5,
                        label=f"Plan {organ} (D_m={D_mean:.1f} Gy, TCP={val:.1%})")
                ax.axvline(D_mean, color=color, lw=0.8, linestyle=":", alpha=0.6)
                tbl.setItem(r, 8, _cell(f"D_media = {D_mean:.1f} Gy"))

            elif "ntcp" in data and p_rb and A.size > 0:
                val   = data["ntcp"]["NTCP"]
                a_par = 1.0 / max(p_rb.n, 1e-6)
                g_eud = geud(A, a_par)
                tbl.setItem(r, 2, _cell(f"NTCP = {val:.2%}"))
                tbl.setItem(r, 3, _cell(f"±{mc.get('NTCP_std',0):.2%}" if mc else "sin σ(A)"))
                tbl.setItem(r, 4, _cell(
                    f"[{mc.get('NTCP_p5',val):.2%} – {mc.get('NTCP_p95',val):.2%}]"
                    if mc else "sin σ(A)"))

                # Curva teórica NTCP vs. dosis uniforme [Gy]
                d_max  = max(p_rb.TD50 * 2.5, float(np.mean(A)) * 3.5, 30.0)
                dose_x = np.linspace(0.0, d_max, 400)
                curve  = ntcp_dose_curve(dose_x, p_rb.TD50, p_rb.m)
                ax.plot(dose_x, curve * 100, color=color, lw=2,
                        linestyle="-", label=f"NTCP {organ}")
                # Punto del plan actual: (gEUD, NTCP_real)
                ax.plot(g_eud, val * 100, "s", color=color, ms=8, zorder=5,
                        label=f"Plan {organ} (gEUD={g_eud:.1f} Gy, NTCP={val:.1%})")
                ax.axvline(g_eud, color=color, lw=0.8, linestyle=":", alpha=0.6)
                tbl.setItem(r, 8, _cell(f"TD₅₀ = {p_rb.TD50:.0f} Gy"))

            else:
                for col in [2, 3, 4, 8]:
                    tbl.setItem(r, col, _cell("—"))

        tbl.resizeColumnsToContents()

        ax.set_xlabel("Dosis acumulada uniforme (Gy fotónico equivalente)")
        ax.set_ylabel("Probabilidad (%)")
        ax.set_title(
            "Curvas dosis-respuesta  —  TCP (- -) y NTCP (—)\n"
            "Marcadores individuales representan la dosis/probabilidad del plan actual"
        )
        ax.set_ylim(-2, 105)
        ax.legend(fontsize=7, framealpha=0.9, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), ncol=1)
        ax.grid(True, alpha=0.3)
        self.canvas_rbc.fig.tight_layout(pad=1.0)
        self.canvas_rbc.fig.subplots_adjust(right=0.75)
        self.canvas_rbc.draw()


    # ── update Tab 4 ─────────────────────────────────────────────────────────
    def _upd_bed(self):
        tbl  = self.tbl_b
        rows = [(org, d.get("organ_type","oar"), d["bed_eqd2"])
                for org, d in self.radiobio_report.get("organs", {}).items()
                if "bed_eqd2" in d]
        tbl.setRowCount(len(rows))
        for r, (organ, otype, b) in enumerate(rows):
            tbl.setItem(r, 0, _cell(organ, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter))
            tbl.setItem(r, 1, _cell(otype))
            tbl.setItem(r, 2, _cell(f"{b.get('alpha_beta_used',0):.1f}"))
            tbl.setItem(r, 3, _cell(f"{b.get('BED_mean', 0):.2f}"))
            tbl.setItem(r, 4, _cell(f"{b.get('BED_D50',  0):.2f}"))
            tbl.setItem(r, 5, _cell(f"{b.get('BED_max',  0):.2f}"))
            tbl.setItem(r, 6, _cell(f"{b.get('EQD2_mean',0):.2f}"))
            tbl.setItem(r, 7, _cell(f"{b.get('EQD2_D50', 0):.2f}"))
            tbl.setItem(r, 8, _cell(f"{b.get('EQD2_max', 0):.2f}"))
        tbl.resizeColumnsToContents()
