"""
ui/dialogs/radiobio.py
======================
Diálogo de análisis radiobiológico para RatMaster.

Se abre desde ResultsDialog → botón "Prob. Dosis-Efecto".
Requiere modo IsoE activo y parámetros isoe_params_by_organ.

Pestañas:
  1. TCP Tumor (HK)      — modelo Hug-Kellerer + Martel (Rubén 2024)
  2. NTCP Piel           — modelo sub-volumen González et al. (2009)
  3. TCP/NTCP clásico    — Poisson + LKB (módulo radiobio existente)

Layout por pestaña:
  QSplitter vertical
  ├── Panel superior (QScrollArea) — selección, parámetros, botón, resultados
  └── Panel inferior (fijo)        — canvas matplotlib con altura garantizada
"""

from __future__ import annotations
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure as MplFigure

from ratmaster.physics.radiobio.tcp_hk import (
    tcp_hk_stats, tcp_hk_dose_curve, mc_tcp_hk_uncertainty,
    MARTEL_PARAMS, DEFAULT_K1, DEFAULT_K2, DEFAULT_K3,
    DEFAULT_D50, DEFAULT_GAMMA, HK_K1_CI, HK_K2_CI, HK_K3_CI,
)
from ratmaster.physics.radiobio.ntcp_skin import (
    skin_dose_stats, ntcp_skin_dose_curve, mc_ntcp_skin_uncertainty,
    ntcp_skin_single_dose_field, ntcp_skin_field_geometry,
    DEFAULT_N0_SKIN, DEFAULT_K_SKIN, DEFAULT_ALPHA_SKIN, DEFAULT_TOP_FRACTION,
    DEFAULT_A_REF_CM2,
)
from ratmaster.physics.radiobio.tcp  import tcp_stats
from ratmaster.physics.radiobio.ntcp import ntcp_stats


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de estilo
# ─────────────────────────────────────────────────────────────────────────────

_HDR_COLOR = "#455A64"
_HDR_TEXT  = "#FFFFFF"
_ROW_EVEN  = "#F5F7FA"
_BORDER    = "#CFD8DC"

_TBL_STYLE = (
    f"QHeaderView::section {{ background:{_HDR_COLOR}; color:{_HDR_TEXT}; "
    f"font-weight:600; padding:4px; border:none; "
    f"border-right:1px solid #546E7A; }}"
    f"QTableWidget {{ alternate-background-color:{_ROW_EVEN}; "
    f"gridline-color:{_BORDER}; }}"
)

_CHART_MIN_H = 280   # altura mínima del panel del gráfico


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dbl(val: float, dec: int = 4) -> QtWidgets.QDoubleSpinBox:
    sb = QtWidgets.QDoubleSpinBox()
    sb.setDecimals(dec)
    sb.setRange(0.0, 1e6)
    sb.setValue(val)
    return sb


def _sel_label(text: str, big: bool = False) -> QtWidgets.QLabel:
    """Label con texto seleccionable (para copiar resultados)."""
    lbl = QtWidgets.QLabel(text)
    lbl.setTextInteractionFlags(
        QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard
    )
    lbl.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
    lbl.setWordWrap(True)
    if big:
        lbl.setStyleSheet("font-size: 13pt; font-weight: bold;")
    return lbl


def _make_table(headers: list[str], rows: list[list]) -> QtWidgets.QTableWidget:
    tbl = QtWidgets.QTableWidget(len(rows), len(headers))
    tbl.setHorizontalHeaderLabels(headers)
    tbl.verticalHeader().setVisible(False)
    tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    tbl.setAlternatingRowColors(True)
    tbl.setStyleSheet(_TBL_STYLE)
    tbl.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Preferred,
    )
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            it = QtWidgets.QTableWidgetItem(str(val))
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            align = (
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
            ) if c == 0 else (
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
            )
            it.setTextAlignment(align)
            tbl.setItem(r, c, it)
    tbl.resizeColumnsToContents()
    tbl.horizontalHeader().setStretchLastSection(True)
    # Altura exacta para no necesitar scroll interno
    h = tbl.horizontalHeader().height() + 4
    for i in range(tbl.rowCount()):
        h += tbl.rowHeight(i)
    tbl.setFixedHeight(h + 4)
    return tbl


def _chart_panel(fig: MplFigure, canvas: FigureCanvasQTAgg) -> QtWidgets.QWidget:
    """Widget contenedor para el gráfico con altura mínima garantizada."""
    panel = QtWidgets.QWidget()
    panel.setMinimumHeight(_CHART_MIN_H)
    panel.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Expanding,
    )
    lay = QtWidgets.QVBoxLayout(panel)
    lay.setContentsMargins(4, 4, 4, 4)
    canvas.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Expanding,
    )
    lay.addWidget(canvas)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Pestaña 1: TCP Tumor — Hug-Kellerer + Martel
# ─────────────────────────────────────────────────────────────────────────────

class _TcpHkTab(QtWidgets.QWidget):
    """
    Layout: QSplitter vertical
      ▲ ScrollArea  — selección de órgano, parámetros, botón, resultados
      ▼ Panel fijo  — gráfico matplotlib (altura mínima _CHART_MIN_H px)
    """

    def __init__(self, parent, iso_vox: dict, sigma_vox: dict):
        super().__init__(parent)
        self.iso_vox   = iso_vox
        self.sigma_vox = sigma_vox

        # ── Canvas (se crea antes para poder pasarlo al panel) ────────
        self.fig  = MplFigure(tight_layout=True)
        self.canv = FigureCanvasQTAgg(self.fig)
        self._draw_empty()

        # ── Splitter principal ────────────────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # Panel superior: scroll con controles y resultados
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        top_widget = self._build_top_panel()
        scroll.setWidget(top_widget)
        splitter.addWidget(scroll)

        # Panel inferior: gráfico
        splitter.addWidget(_chart_panel(self.fig, self.canv))

        # Proporciones iniciales: 45% arriba, 55% abajo
        splitter.setSizes([400, 350])

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

    # ── Construcción del panel superior ──────────────────────────────

    def _build_top_panel(self) -> QtWidgets.QWidget:
        w   = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(8, 8, 8, 4)

        # Selección de órgano
        row_org = QtWidgets.QHBoxLayout()
        row_org.addWidget(QtWidgets.QLabel("<b>Órgano tumoral:</b>"))
        self.organ_cb = QtWidgets.QComboBox()
        organs = sorted(self.iso_vox.keys())
        self.organ_cb.addItems(organs)
        for i, o in enumerate(organs):
            if "tumor" in o.lower():
                self.organ_cb.setCurrentIndex(i)
                break
        row_org.addWidget(self.organ_cb)
        row_org.addStretch()
        lay.addLayout(row_org)

        # GroupBox parámetros
        gb = QtWidgets.QGroupBox("Parámetros del modelo")
        form = QtWidgets.QFormLayout(gb)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)

        # Seguimiento Martel
        self.followup_cb = QtWidgets.QComboBox()
        for m, (d50, g, _) in MARTEL_PARAMS.items():
            self.followup_cb.addItem(
                f"{m} meses  (D50={d50} Gy, γ={g})", userData=(d50, g)
            )
        self.followup_cb.setCurrentIndex(1)
        form.addRow("Seguimiento Martel:", self.followup_cb)

        # k1, k2, k3
        hk_row = QtWidgets.QHBoxLayout()
        self.sb_k1 = _dbl(DEFAULT_K1, 4); self.sb_k1.setRange(0.01, 100.0)
        self.sb_k2 = _dbl(DEFAULT_K2, 4); self.sb_k2.setRange(0.01, 200.0)
        self.sb_k3 = _dbl(DEFAULT_K3, 4); self.sb_k3.setRange(0.001, 10.0)
        for lbl, sb in (("k₁:", self.sb_k1), ("k₂:", self.sb_k2), ("k₃:", self.sb_k3)):
            hk_row.addWidget(QtWidgets.QLabel(lbl))
            hk_row.addWidget(sb)
        hk_row.addWidget(QtWidgets.QLabel("[Gy⁻¹]"))
        hk_row.addStretch()
        form.addRow("HK (H460):", hk_row)

        ic_lbl = QtWidgets.QLabel(
            f"IC 95%: k₁ [{HK_K1_CI[0]}, {HK_K1_CI[1]}]  "
            f"k₂ [{HK_K2_CI[0]}, {HK_K2_CI[1]}]  "
            f"k₃ [{HK_K3_CI[0]}, {HK_K3_CI[1]}]"
        )
        ic_lbl.setStyleSheet("color:#607D8B; font-size:9pt;")
        form.addRow("", ic_lbl)

        # D50, γ
        martel_row = QtWidgets.QHBoxLayout()
        self.sb_d50   = _dbl(DEFAULT_D50,   2); self.sb_d50.setRange(1.0, 300.0)
        self.sb_gamma = _dbl(DEFAULT_GAMMA,  3); self.sb_gamma.setRange(0.1, 20.0)
        martel_row.addWidget(QtWidgets.QLabel("D50:"))
        martel_row.addWidget(self.sb_d50)
        martel_row.addWidget(QtWidgets.QLabel("Gy   γ:"))
        martel_row.addWidget(self.sb_gamma)
        martel_row.addStretch()
        form.addRow("Martel (NSCLC):", martel_row)

        # Muestras MC
        self.sb_mc = QtWidgets.QSpinBox()
        self.sb_mc.setRange(100, 5000); self.sb_mc.setValue(500)
        self.sb_mc.setSingleStep(100)
        form.addRow("Muestras MC:", self.sb_mc)

        lay.addWidget(gb)

        # Botón
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("Calcular TCP (HK)")
        self.btn_calc.setStyleSheet(
            "QPushButton{background:#1565C0;color:white;font-weight:600;"
            "padding:6px 20px;border-radius:4px;}"
            "QPushButton:hover{background:#1976D2;}"
        )
        btn_row.addWidget(self.btn_calc)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Resultados
        res_gb  = QtWidgets.QGroupBox("Resultado")
        res_lay = QtWidgets.QVBoxLayout(res_gb)
        res_lay.setSpacing(3)

        self.lbl_tcp  = _sel_label("TCP (HK): —", big=True)
        self.lbl_tcp.setStyleSheet(
            "font-size:14pt; font-weight:bold; color:#1565C0;"
        )
        self.lbl_mc   = _sel_label("IC 90% MC: —")
        self.lbl_deq  = _sel_label("EQD2 media (2 Gy/fx): —")
        self.lbl_diso = _sel_label("Dosis IsoE media: —")
        self.lbl_nvox = _sel_label("Vóxeles: —")
        self.lbl_nvox.setStyleSheet("color:#607D8B; font-size:9pt;")

        for lbl in (self.lbl_tcp, self.lbl_mc, self.lbl_deq,
                    self.lbl_diso, self.lbl_nvox):
            res_lay.addWidget(lbl)

        lay.addWidget(res_gb)
        lay.addStretch()

        # Señales
        self.followup_cb.currentIndexChanged.connect(self._on_followup)
        self.btn_calc.clicked.connect(self._calc)

        return w

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_followup(self, _):
        d50, g = self.followup_cb.currentData()
        self.sb_d50.setValue(d50)
        self.sb_gamma.setValue(g)

    def _calc(self):
        organ  = self.organ_cb.currentText()
        A      = np.asarray(self.iso_vox.get(organ, []), float)
        if A.size == 0:
            QtWidgets.QMessageBox.warning(
                self, "Sin datos",
                f"No hay datos de dosis isoefectiva para '{organ}'."
            )
            return

        sA = np.asarray(self.sigma_vox.get(organ, [0.0] * A.size), float)
        if sA.size != A.size:
            sA = np.zeros_like(A)

        k1, k2, k3 = self.sb_k1.value(), self.sb_k2.value(), self.sb_k3.value()
        D50  = self.sb_d50.value()
        gam  = self.sb_gamma.value()
        N_mc = self.sb_mc.value()

        st = tcp_hk_stats(A, k1, k2, k3, D50, gam)
        mc = mc_tcp_hk_uncertainty(A, sA, k1, k2, k3, D50, gam, N_samples=N_mc)

        v = st["TCP_HK"]
        self.lbl_tcp.setText(f"TCP (HK): {v:.4f}  ({v*100:.1f}%)")
        self.lbl_mc.setText(
            f"IC 90% MC: [{mc['TCP_p5']:.4f}, {mc['TCP_p95']:.4f}]"
            f"   σ = {mc['TCP_std']:.4f}"
        )
        self.lbl_deq.setText(
            f"EQD2 media (2 Gy/fx): {st['D_eq_mean']:.2f} Gy"
        )
        self.lbl_diso.setText(
            f"Dosis IsoE media: {st['D_iso_mean']:.2f} Gy_eq"
        )
        self.lbl_nvox.setText(f"Vóxeles: {st['N_voxels']:,}")

        self._plot(A, st, k1, k2, k3, D50, gam, organ)

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlabel("Dosis IsoE BNCT [Gy_eq]", fontsize=9)
        ax.set_ylabel("TCP (%)", fontsize=9)
        ax.set_title("Curva dosis-respuesta TCP (HK + Martel)", fontsize=9)
        ax.set_xlim(0, 100); ax.set_ylim(0, 105)
        ax.grid(True, color="#E0E0E0", lw=0.7)
        ax.text(0.5, 0.5, "Presioná «Calcular TCP» para ver la curva",
                ha="center", va="center", transform=ax.transAxes,
                color="#90A4AE", fontsize=10)
        self.canv.draw()

    def _plot(self, A, st, k1, k2, k3, D50, gam, organ):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        d_max   = max(float(np.max(A)) * 1.6, 50.0)
        d_curve = np.linspace(0.0, d_max, 400)
        tcp_c   = tcp_hk_dose_curve(d_curve, k1, k2, k3, D50, gam)

        ax.plot(d_curve, tcp_c * 100, color="#1565C0", lw=2,
                label=f"TCP HK  (D50={D50:.0f} Gy, γ={gam:.1f})")

        v      = st["TCP_HK"]
        d_mean = st["D_iso_mean"]
        ax.plot(d_mean, v * 100, "o", color="#E53935", ms=9, zorder=5,
                label=f"Plan: D_mean={d_mean:.1f} Gy_eq  →  TCP={v*100:.1f}%")

        ax.axhline(50, color="#757575", ls="--", lw=0.8, alpha=0.6, label="50%")
        ax.axhline(80, color="#43A047", ls="--", lw=0.8, alpha=0.6, label="80%")
        ax.axvline(D50, color="#FF8F00", ls=":", lw=0.9, alpha=0.7,
                   label=f"D50 = {D50:.0f} Gy")

        ax.set_xlabel("Dosis IsoE BNCT (fracción única) [Gy_eq]", fontsize=9)
        ax.set_ylabel("TCP (%)", fontsize=9)
        ax.set_title(f"TCP — HK + Martel — {organ}", fontsize=9)
        ax.set_xlim(0, d_max); ax.set_ylim(0, 105)
        ax.grid(True, color="#E0E0E0", lw=0.7)
        ax.legend(fontsize=8, framealpha=0.88, loc="lower right")
        self.fig.tight_layout(pad=1.2)
        self.canv.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Pestaña 2: NTCP Piel — González et al. 2009
# ─────────────────────────────────────────────────────────────────────────────

class _NtcpSkinTab(QtWidgets.QWidget):
    """
    Layout: QSplitter vertical
      ▲ ScrollArea  — selección, parámetros, botón, resultados + tabla FOM
      ▼ Panel fijo  — gráfico matplotlib
    """

    _SKIN_KW = ("piel", "skin", "derm")

    def __init__(self, parent, iso_vox: dict, sigma_vox: dict):
        super().__init__(parent)
        self.iso_vox   = iso_vox
        self.sigma_vox = sigma_vox
        self._last_fom_tbl: QtWidgets.QTableWidget | None = None

        self.fig  = MplFigure(tight_layout=True)
        self.canv = FigureCanvasQTAgg(self.fig)
        self._draw_empty()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_top_panel())
        splitter.addWidget(scroll)

        splitter.addWidget(_chart_panel(self.fig, self.canv))
        splitter.setSizes([420, 330])

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

    def _build_top_panel(self) -> QtWidgets.QWidget:
        w   = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(8, 8, 8, 4)

        # Selección de órgano
        row_org = QtWidgets.QHBoxLayout()
        row_org.addWidget(QtWidgets.QLabel("<b>Órgano piel:</b>"))
        self.organ_cb = QtWidgets.QComboBox()
        organs = sorted(self.iso_vox.keys())
        self.organ_cb.addItems(organs)
        for i, o in enumerate(organs):
            if any(k in o.lower() for k in self._SKIN_KW):
                self.organ_cb.setCurrentIndex(i)
                break
        row_org.addWidget(self.organ_cb)
        row_org.addStretch()
        lay.addLayout(row_org)

        # Parámetros
        gb   = QtWidgets.QGroupBox("Parámetros  [González et al. 2009, Ec. 2.1]")
        form = QtWidgets.QFormLayout(gb)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)

        self.sb_N0    = _dbl(DEFAULT_N0_SKIN,    3); self.sb_N0.setRange(0.001, 100.0)
        self.sb_k     = _dbl(DEFAULT_K_SKIN,     3); self.sb_k.setRange(0.01, 5.0)
        self.sb_alpha = _dbl(DEFAULT_ALPHA_SKIN,  4); self.sb_alpha.setRange(0.001, 5.0)
        form.addRow("N₀:", self.sb_N0)
        form.addRow("k:",  self.sb_k)
        form.addRow("α [Gy⁻¹]:", self.sb_alpha)

        note = QtWidgets.QLabel(
            "Ajustados a tolerancia de piel, fracción única "
            "(Ellis 1968, Hopewell 1990)."
        )
        note.setStyleSheet("color:#607D8B; font-size:9pt;")
        note.setWordWrap(True)
        form.addRow("", note)

        self.sb_topf = _dbl(DEFAULT_TOP_FRACTION * 100, 1)
        self.sb_topf.setRange(1.0, 100.0)
        self.sb_topf.setSuffix("%")
        self.sb_topf.setToolTip(
            "Fracción de vóxeles de mayor dosis para D_top_mean y PEUD\n"
            "(proxy de 100 cm² del paper)."
        )
        form.addRow("Top % vóxeles (≈ 100 cm²):", self.sb_topf)

        self.sb_mc = QtWidgets.QSpinBox()
        self.sb_mc.setRange(100, 5000); self.sb_mc.setValue(500)
        self.sb_mc.setSingleStep(100)
        form.addRow("Muestras MC:", self.sb_mc)

        lay.addWidget(gb)

        # ── Grupo: NTCP por geometría de campo real ───────────────────
        # N0,k,α fueron calibrados con ν = fracción de un área de
        # referencia clínica (100 cm²). La fracción de vóxeles/volumen
        # de la piel SEGMENTADA del rat (panel de arriba) NO es esa
        # misma escala — para un campo de irradiación geométrico (p.ej.
        # el corte cilíndrico parcial calculado por geometría), usar
        # este panel en su lugar.
        gb_field = QtWidgets.QGroupBox(
            "NTCP por geometría de campo real  [recomendado p/ campo parcial]"
        )
        form_f = QtWidgets.QFormLayout(gb_field)
        form_f.setHorizontalSpacing(12)
        form_f.setVerticalSpacing(6)

        note_f = QtWidgets.QLabel(
            "Usar cuando el campo irradiado es una región geométrica "
            "(ej. media superficie cilíndrica) y NO toda la piel "
            "segmentada del rat. ν se calcula como fracción del área "
            "de referencia (100 cm², González 2009), no como fracción "
            "del volumen de piel segmentado."
        )
        note_f.setWordWrap(True)
        note_f.setStyleSheet("color:#607D8B; font-size:9pt;")
        form_f.addRow(note_f)

        self.sb_Dfield = _dbl(28.5, 2)
        self.sb_Dfield.setRange(0.0, 200.0)
        self.sb_Dfield.setToolTip(
            "Dosis isoefectiva representativa del campo [Gy_eq]\n"
            "(ej. D_max, o el valor escalado por Cohen-Kerrich)."
        )
        form_f.addRow("D_campo [Gy_eq]:", self.sb_Dfield)

        self.sb_Afield = _dbl(20.9, 2)
        self.sb_Afield.setRange(0.001, 10000.0)
        self.sb_Afield.setToolTip(
            "Área geométrica real del campo irradiado [cm²]\n"
            "(ej. A = π·r·L para el corte cilíndrico parcial)."
        )
        form_f.addRow("A_campo [cm²]:", self.sb_Afield)

        self.sb_Aref = _dbl(DEFAULT_A_REF_CM2, 1)
        self.sb_Aref.setRange(1.0, 10000.0)
        self.sb_Aref.setToolTip(
            "Área de referencia de calibración del modelo\n"
            "(100 cm² en González et al. 2009 — no cambiar salvo\n"
            "recalibración de N0, k, α)."
        )
        form_f.addRow("A_ref [cm²]:", self.sb_Aref)

        self.btn_calc_field = QtWidgets.QPushButton("Calcular NTCP de campo")
        self.btn_calc_field.setStyleSheet(
            "QPushButton{background:#4527A0;color:white;font-weight:600;"
            "padding:6px 20px;border-radius:4px;}"
            "QPushButton:hover{background:#512DA8;}"
        )
        form_f.addRow(self.btn_calc_field)

        self.lbl_ntcp_field = _sel_label("NTCP (campo): —", big=True)
        self.lbl_ntcp_field.setStyleSheet(
            "font-size:13pt; font-weight:bold; color:#4527A0;"
        )
        self.lbl_nu_field = _sel_label("ν_campo = A_campo / A_ref: —")
        form_f.addRow(self.lbl_ntcp_field)
        form_f.addRow(self.lbl_nu_field)

        lay.addWidget(gb_field)
        self.btn_calc_field.clicked.connect(self._calc_field)
        # ────────────────────────────────────────────────────────────

        # Botón
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("Calcular NTCP Piel")
        self.btn_calc.setStyleSheet(
            "QPushButton{background:#6A1B9A;color:white;font-weight:600;"
            "padding:6px 20px;border-radius:4px;}"
            "QPushButton:hover{background:#7B1FA2;}"
        )
        btn_row.addWidget(self.btn_calc)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Resultados numéricos
        res_gb  = QtWidgets.QGroupBox("Resultado")
        res_lay = QtWidgets.QVBoxLayout(res_gb)
        res_lay.setSpacing(3)

        self.lbl_ntcp  = _sel_label("NTCP (piel): —", big=True)
        self.lbl_ntcp.setStyleSheet(
            "font-size:14pt; font-weight:bold; color:#6A1B9A;"
        )
        self.lbl_mc   = _sel_label("IC 90% MC: —")
        self.lbl_dmax = _sel_label("D_max: —")
        self.lbl_dtop = _sel_label("D_top_mean: —")
        self.lbl_peud = _sel_label("PEUD_top: —")
        self.lbl_frac = _sel_label("Fracción ≥15/18/20 Gy-Eq: —")
        self.lbl_nvox = _sel_label("Vóxeles: —")
        self.lbl_nvox.setStyleSheet("color:#607D8B; font-size:9pt;")

        for lbl in (self.lbl_ntcp, self.lbl_mc, self.lbl_dmax,
                    self.lbl_dtop, self.lbl_peud, self.lbl_frac, self.lbl_nvox):
            res_lay.addWidget(lbl)

        # Tabla FOM (se inserta dinámicamente después del cálculo)
        self._fom_container = QtWidgets.QWidget()
        self._fom_lay       = QtWidgets.QVBoxLayout(self._fom_container)
        self._fom_lay.setContentsMargins(0, 4, 0, 0)
        res_lay.addWidget(self._fom_container)

        lay.addWidget(res_gb)
        lay.addStretch()

        self.btn_calc.clicked.connect(self._calc)
        return w

    def _calc(self):
        organ = self.organ_cb.currentText()
        A     = np.asarray(self.iso_vox.get(organ, []), float)
        if A.size == 0:
            QtWidgets.QMessageBox.warning(
                self, "Sin datos",
                f"No hay datos de dosis isoefectiva para '{organ}'."
            )
            return

        sA = np.asarray(self.sigma_vox.get(organ, [0.0] * A.size), float)
        if sA.size != A.size:
            sA = np.zeros_like(A)

        N0    = self.sb_N0.value()
        k     = self.sb_k.value()
        alpha = self.sb_alpha.value()
        tf    = self.sb_topf.value() / 100.0
        N_mc  = self.sb_mc.value()

        st = skin_dose_stats(A, tf, N0, k, alpha)
        mc = mc_ntcp_skin_uncertainty(A, sA, N0, k, alpha, N_samples=N_mc)

        v = st["NTCP_skin"]
        self.lbl_ntcp.setText(f"NTCP (piel): {v:.4f}  ({v*100:.1f}%)")
        self.lbl_mc.setText(
            f"IC 90% MC: [{mc['NTCP_p5']:.4f}, {mc['NTCP_p95']:.4f}]"
            f"   σ = {mc['NTCP_std']:.4f}"
        )
        self.lbl_dmax.setText(f"D_max: {st['D_max_Gy']:.2f} Gy-Eq")
        self.lbl_dtop.setText(
            f"D_top_mean ({tf*100:.0f}% vóx): {st['D_top_mean_Gy']:.2f} Gy-Eq"
            "  ← proxy D100_mean"
        )
        self.lbl_peud.setText(
            f"PEUD_top ({tf*100:.0f}% vóx): {st['PEUD_top_Gy']:.2f} Gy-Eq"
            "  ← proxy PEUD100"
        )
        self.lbl_frac.setText(
            f"≥15 Gy: {st['frac_above_15']*100:.1f}%  |  "
            f"≥18 Gy: {st['frac_above_18']*100:.1f}%  |  "
            f"≥20 Gy: {st['frac_above_20']*100:.1f}%"
        )
        self.lbl_nvox.setText(f"Vóxeles: {st['N_voxels']:,}")

        # Actualizar tabla FOM
        if self._last_fom_tbl is not None:
            self._last_fom_tbl.setParent(None)
            self._last_fom_tbl.deleteLater()

        fom_rows = [
            ["D_max",           f"{st['D_max_Gy']:.3f}",      "Gy-Eq", "Dosis puntual máxima"],
            ["D_top_mean",      f"{st['D_top_mean_Gy']:.3f}",  "Gy-Eq", f"Media top {tf*100:.0f}% (≈ D100_mean)"],
            ["PEUD_top",        f"{st['PEUD_top_Gy']:.3f}",    "Gy-Eq", f"PEUD top {tf*100:.0f}% (≈ PEUD100)"],
            ["NTCP_skin",       f"{v:.4f}",                    "",       "Probabilidad de complicación"],
            ["Frac. ≥ 15 Gy",  f"{st['frac_above_15']*100:.1f}%", "",  ""],
            ["Frac. ≥ 18 Gy",  f"{st['frac_above_18']*100:.1f}%", "",  "Umbral eritema/úlcera"],
            ["Frac. ≥ 20 Gy",  f"{st['frac_above_20']*100:.1f}%", "",  ""],
        ]
        tbl = _make_table(
            ["Figura de mérito", "Valor", "Unidad", "Nota"], fom_rows
        )
        self._last_fom_tbl = tbl
        self._fom_lay.addWidget(tbl)

        self._plot(A, st, N0, k, alpha, organ)

    def _calc_field(self):
        D_field = self.sb_Dfield.value()
        A_field = self.sb_Afield.value()
        A_ref   = self.sb_Aref.value()
        N0      = self.sb_N0.value()
        k       = self.sb_k.value()
        alpha   = self.sb_alpha.value()

        r = ntcp_skin_single_dose_field(D_field, A_field, A_ref, N0, k, alpha)
        v = r["NTCP_field"]
        self.lbl_ntcp_field.setText(
            f"NTCP (campo): {v:.4f}  ({v*100:.1f}%)"
        )
        self.lbl_nu_field.setText(
            f"ν_campo = A_campo / A_ref = {A_field:.2f} / {A_ref:.1f} "
            f"= {r['nu_field']:.4f}"
        )

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlabel("Dosis IsoE BNCT [Gy-Eq]", fontsize=9)
        ax.set_ylabel("NTCP piel (%)", fontsize=9)
        ax.set_title("Curva NTCP piel (González et al. 2009)", fontsize=9)
        ax.set_xlim(0, 40); ax.set_ylim(0, 105)
        ax.grid(True, color="#E0E0E0", lw=0.7)
        ax.text(0.5, 0.5, "Presioná «Calcular NTCP Piel» para ver la curva",
                ha="center", va="center", transform=ax.transAxes,
                color="#90A4AE", fontsize=10)
        self.canv.draw()

    def _plot(self, A, st, N0, k, alpha, organ):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        d_max_ax = max(float(np.max(A)) * 1.6, 35.0)
        d_curve  = np.linspace(0.0, d_max_ax, 300)
        # Curva de referencia: NTCP si TODA la piel estuviera a esa dosis
        # uniformemente (ν=1). Ya NO depende de N_vox.
        ntcp_c   = ntcp_skin_dose_curve(d_curve, N0, k, alpha)

        ax.plot(d_curve, ntcp_c * 100, color="#6A1B9A", lw=2,
                label="Curva de referencia (piel 100% a dosis D)")

        v    = st["NTCP_skin"]
        dmax = st["D_max_Gy"]
        dmean = st["D_mean_Gy"]

        # Punto del plan real: NTCP de la distribución NO uniforme completa.
        # Se ubica en D_mean (referencia de "carga" típica), no en D_max,
        # porque la curva asume dosis uniforme y el plan real no lo es.
        ax.plot(dmean, v * 100, "D", color="#E53935", ms=10, zorder=5,
                label=f"Plan real (no uniforme): NTCP={v*100:.1f}%")
        # Línea vertical punteada en D_max para referencia visual
        if dmax <= d_max_ax:
            ax.axvline(dmax, color="#E53935", ls=":", lw=1.0, alpha=0.5)
            ax.annotate(f"D_max={dmax:.1f}", xy=(dmax, 2), fontsize=7.5,
                        color="#B71C1C", rotation=90, va="bottom", ha="right")

        # Bandas de referencia
        if d_max_ax > 15:
            ax.axvspan(15, min(18, d_max_ax), alpha=0.09, color="#FFA000",
                       label="Umbral eritema/úlcera (15–18 Gy)")
        if d_max_ax > 18:
            ax.axvspan(18, min(20, d_max_ax), alpha=0.13, color="#E53935")

        ax.set_xlabel("Dosis IsoE BNCT (fracción única) [Gy-Eq]", fontsize=9)
        ax.set_ylabel("NTCP piel (%)", fontsize=9)
        ax.set_title(f"NTCP piel — González et al. 2009 — {organ}", fontsize=9)
        ax.set_xlim(0, d_max_ax); ax.set_ylim(0, 105)
        ax.grid(True, color="#E0E0E0", lw=0.7)
        ax.legend(fontsize=7.5, framealpha=0.88, loc="upper left")

        note = (
            "Nota: la curva asume toda la piel a dosis uniforme D. "
            "El punto del plan usa la distribución real (no uniforme) "
            "y por eso, en general, NO cae sobre la curva."
        )
        ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=7, color="#607D8B", wrap=True)

        self.fig.tight_layout(pad=1.2, rect=(0, 0.05, 1, 1))
        self.canv.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Pestaña 3: TCP/NTCP clásico — Poisson + LKB
# ─────────────────────────────────────────────────────────────────────────────

class _ClassicTab(QtWidgets.QWidget):
    """
    Layout simple: scroll con todo el contenido.
    No tiene gráfico, así que no necesita splitter.
    """

    def __init__(self, parent, iso_vox: dict, sigma_vox: dict,
                 isoe_params_by_organ: dict):
        super().__init__(parent)
        self.iso_vox      = iso_vox
        self.sigma_vox    = sigma_vox
        self.isoe_params  = isoe_params_by_organ
        self._last_tbl: QtWidgets.QTableWidget | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        note = QtWidgets.QLabel(
            "<b>Modelos clásicos:</b>  TCP = Poisson (Webb &amp; Nahum 1993)"
            "  |  NTCP = LKB (Kutcher &amp; Burman 1989).<br>"
            "Parámetros por defecto orientativos — ajustar según tejido."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#455A64; font-size:9pt;")
        root.addWidget(note)

        # Parámetros por órgano (en scroll)
        gb  = QtWidgets.QGroupBox("Parámetros por órgano")
        gbl = QtWidgets.QVBoxLayout(gb)

        sc     = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setMaximumHeight(200)
        sc.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner  = QtWidgets.QWidget()
        f_lay  = QtWidgets.QFormLayout(inner)
        f_lay.setHorizontalSpacing(8)
        f_lay.setVerticalSpacing(4)

        organs = sorted(iso_vox.keys())
        self.organ_params: dict[str, dict] = {}

        for org in organs:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(4)

            type_cb = QtWidgets.QComboBox()
            type_cb.addItems(["OAR", "Tumor"])
            if "tumor" in org.lower():
                type_cb.setCurrentIndex(1)

            sb_N0   = _dbl(1e7, 3); sb_N0.setRange(1.0, 1e12)
            sb_td50 = _dbl(60.0, 1); sb_td50.setRange(1.0, 200.0)
            sb_m    = _dbl(0.15, 3); sb_m.setRange(0.01, 1.0)
            sb_n    = _dbl(0.10, 3); sb_n.setRange(0.001, 1.0)
            sb_ab   = _dbl(3.0, 2);  sb_ab.setRange(0.1, 30.0)

            for lbl_txt, sb in (
                ("Tipo:", type_cb), ("N₀:", sb_N0), ("TD50:", sb_td50),
                ("m:", sb_m), ("n:", sb_n), ("α/β:", sb_ab),
            ):
                row.addWidget(QtWidgets.QLabel(lbl_txt))
                row.addWidget(sb)
            row.addStretch()

            self.organ_params[org] = {
                "type_cb": type_cb, "sb_N0": sb_N0,
                "sb_td50": sb_td50, "sb_m": sb_m,
                "sb_n": sb_n, "sb_ab": sb_ab,
            }
            w = QtWidgets.QWidget(); w.setLayout(row)
            f_lay.addRow(f"<b>{org}</b>:", w)

        sc.setWidget(inner)
        gbl.addWidget(sc)
        root.addWidget(gb)

        # Botón
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("Calcular TCP / NTCP clásico")
        self.btn_calc.setStyleSheet(
            "QPushButton{background:#2E7D32;color:white;font-weight:600;"
            "padding:6px 20px;border-radius:4px;}"
            "QPushButton:hover{background:#388E3C;}"
        )
        btn_row.addWidget(self.btn_calc)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Contenedor para tabla de resultados
        self._res_container = QtWidgets.QWidget()
        self._res_lay       = QtWidgets.QVBoxLayout(self._res_container)
        self._res_lay.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._res_container)
        root.addStretch()

        self.btn_calc.clicked.connect(self._calc)

    def _calc(self):
        rows = []
        for org, w in self.organ_params.items():
            A = np.asarray(self.iso_vox.get(org, []), float)
            if A.size == 0:
                continue
            iso_p = self.isoe_params.get(org)
            if iso_p is None:
                continue

            aR, bR, GR = iso_p.aR, iso_p.bR, iso_p.GR
            t = w["type_cb"].currentText()

            if t == "Tumor":
                s = tcp_stats(A, aR, bR, GR, w["sb_N0"].value())
                rows.append([org, "Tumor",
                              f"{s['TCP']:.4f}", "—",
                              f"{s['D_mean_Gy']:.2f}", f"{s['mean_S']:.4f}"])
            else:
                s = ntcp_stats(A, w["sb_td50"].value(),
                               w["sb_m"].value(), w["sb_n"].value(),
                               model="lkb")
                rows.append([org, "OAR",
                              "—", f"{s['NTCP']:.4f}",
                              f"{s['D_mean_Gy']:.2f}", f"{s['gEUD_Gy']:.2f}"])

        # Limpiar resultado anterior
        if self._last_tbl is not None:
            self._last_tbl.setParent(None)
            self._last_tbl.deleteLater()
            self._last_tbl = None

        if not rows:
            lbl = QtWidgets.QLabel("Sin resultados — revisar parámetros.")
            self._res_lay.addWidget(lbl)
            return

        tbl = _make_table(
            ["Órgano", "Tipo", "TCP", "NTCP", "D_mean [Gy_eq]", "S_mean / gEUD"],
            rows,
        )
        self._last_tbl = tbl
        self._res_lay.addWidget(tbl)


# ─────────────────────────────────────────────────────────────────────────────
# Diálogo principal
# ─────────────────────────────────────────────────────────────────────────────

class RadiobioDialog(QtWidgets.QDialog):
    """
    Ventana de análisis radiobiológico — TCP, NTCP y supervivencia.

    Llamada desde ResultsDialog._open_radiobio_dialog().

    Args:
        report               : dict con IsoVoxel y SigmaIsoVoxel.
        isoe_params_by_organ : dict {organ_key: IsoEParams} con aR, bR, GR.
    """

    def __init__(self, parent, report: dict, isoe_params_by_organ: dict):
        super().__init__(parent)
        self.setWindowTitle("Análisis Radiobiológico — TCP / NTCP")
        self.resize(920, 760)
        self.setMinimumSize(740, 600)

        iso_vox   = report.get("IsoVoxel", {})
        sigma_vox = report.get("SigmaIsoVoxel", {})

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(4)

        if not iso_vox:
            root.addWidget(QtWidgets.QLabel(
                "No hay datos de dosis isoefectiva.\n"
                "Calculá en modo IsoE primero."
            ))
            root.addWidget(
                QtWidgets.QPushButton("Cerrar", clicked=self.accept)
            )
            return

        # Encabezado
        n_org = len(iso_vox)
        n_vox = sum(len(v) for v in iso_vox.values())
        hdr = QtWidgets.QLabel(
            f"<b>Análisis Radiobiológico BNCT</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;{n_org} órganos"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;{n_vox:,} vóxeles totales"
        )
        hdr.setStyleSheet("font-size:10pt; color:#263238;")
        root.addWidget(hdr)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet("color:#B0BEC5;")
        root.addWidget(sep)

        # Pestañas — cada tab es directamente el widget (sin QScrollArea extra
        # en el nivel del tab, porque cada tab ya gestiona su propio scroll)
        tabs = QtWidgets.QTabWidget()
        tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        tabs.addTab(
            _TcpHkTab(self, iso_vox, sigma_vox),
            "TCP Tumor  (HK + Martel)",
        )
        tabs.addTab(
            _NtcpSkinTab(self, iso_vox, sigma_vox),
            "NTCP Piel  (González 2009)",
        )
        tabs.addTab(
            _ClassicTab(self, iso_vox, sigma_vox, isoe_params_by_organ),
            "TCP / NTCP Clásico  (Poisson + LKB)",
        )

        root.addWidget(tabs, stretch=1)

        # Referencias
        ref = QtWidgets.QLabel(
            "<small><b>Refs:</b> "
            "González et al. 2009 (NTCP piel) · "
            "Martel et al. 1999 (D50/γ NSCLC) · "
            "González &amp; Santa Cruz 2012 (HK para BNCT) · "
            "Park et al. 2008 (HK H460) · "
            "Kutcher &amp; Burman 1989 (LKB) · "
            "Webb &amp; Nahum 1993 (TCP Poisson)"
            "</small>"
        )
        ref.setWordWrap(True)
        ref.setStyleSheet("color:#607D8B;")
        root.addWidget(ref)

        # Botón cerrar
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)
