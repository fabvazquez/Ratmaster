"""
ui/dialogs/radiobio.py
======================
Diálogo de análisis radiobiológico integrado en ResultsDialog.

Pestañas:
    1. Configuración — asignar tipo de órgano y parámetros TCP/NTCP por órgano
    2. TCP / NTCP    — tabla de resultados con incertidumbre MC
    3. Supervivencia — DVH de supervivencia S(A) por órgano
    4. BED / EQD2   — BED y EQD2 por órgano (comparación con fotones)
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.ui.canvas import MplCanvas
from ratmaster.physics.dose_utils import build_dvh, dvh_extend_to_zero
from ratmaster.physics.radiobio.models import (
    compute_radiobio_report,
    RadiobioOrganParams,
    TISSUE_DEFAULTS,
)
from ratmaster.physics.radiobio.survival import survival_voxel


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de layout
# ─────────────────────────────────────────────────────────────────────────────

_ORGAN_COLORS = [
    "#E53935","#8E24AA","#1E88E5","#00ACC1","#43A047",
    "#FB8C00","#6D4C41","#546E7A","#F06292","#AED581",
]
_NTCP_MODEL_OPTIONS = ["lkb", "logistic"]


# ─────────────────────────────────────────────────────────────────────────────
# Diálogo principal
# ─────────────────────────────────────────────────────────────────────────────

class RadiobioDialog(QtWidgets.QDialog):
    """
    Diálogo de análisis radiobiológico. Recibe el reporte IsoE completo
    (que ya contiene IsoVoxel y SigmaIsoVoxel) y los IsoEParams por órgano
    (para obtener aR, bR, GR por tejido).
    """

    def __init__(self, parent, report: dict, isoe_params_by_organ: dict):
        super().__init__(parent)
        self.setWindowTitle("Análisis Radiobiológico — TCP / NTCP / BED")
        self.resize(1000, 680)
        self.report               = report
        self.isoe_params_by_organ = isoe_params_by_organ
        self.radiobio_report: dict | None = None

        # Órganos disponibles en el reporte IsoE
        self.organs = sorted(report.get("IsoVoxel", {}).keys())

        # Parámetros radiobiológicos por órgano (editables en Tab 1)
        self.rb_params: dict[str, RadiobioOrganParams] = {}
        self._init_default_params()

        # Widgets de configuración (fila por órgano)
        self._config_rows: dict[str, dict] = {}

        self._build_ui()

    # ─── Parámetros por defecto ───────────────────────────────────────────
    def _init_default_params(self):
        """Asigna parámetros por defecto a cada órgano según heurística."""
        for organ in self.organs:
            key_lower = organ.lower()
            if any(h in key_lower for h in ["tumor","gtv","ctv","melanoma","glioma"]):
                p = RadiobioOrganParams(organ_type="tumor", N0=1e7, alpha_beta=10.0)
            elif any(h in key_lower for h in ["brain","cerebro"]):
                p = RadiobioOrganParams(**vars(TISSUE_DEFAULTS.get("normal_brain",
                    RadiobioOrganParams())))
            elif "spinal" in key_lower or "medula" in key_lower:
                p = RadiobioOrganParams(**vars(TISSUE_DEFAULTS.get("spinal_cord",
                    RadiobioOrganParams())))
            elif "skin" in key_lower or "piel" in key_lower:
                p = RadiobioOrganParams(**vars(TISSUE_DEFAULTS.get("skin",
                    RadiobioOrganParams())))
            elif "lung" in key_lower or "pulmon" in key_lower:
                p = RadiobioOrganParams(**vars(TISSUE_DEFAULTS.get("lung",
                    RadiobioOrganParams())))
            else:
                p = RadiobioOrganParams()   # OAR genérico
            self.rb_params[organ] = p

    # ─── Construcción UI ──────────────────────────────────────────────────
    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(tabs)

        tabs.addTab(self._build_config_tab(),     "⚙  Configuración")
        tabs.addTab(self._build_results_tab(),    "📊  TCP / NTCP")
        tabs.addTab(self._build_survival_tab(),   "🧬  Supervivencia")
        tabs.addTab(self._build_bed_tab(),        "⚡  BED / EQD2")

        # Botones
        btn_row = QtWidgets.QHBoxLayout()
        btn_calc  = QtWidgets.QPushButton("▶  Calcular")
        btn_calc.setDefault(True)
        btn_close = QtWidgets.QPushButton("Cerrar")
        btn_row.addWidget(btn_calc)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)

        btn_calc.clicked.connect(self._run_calculation)
        btn_close.clicked.connect(self.accept)

        self.tabs = tabs

    # ── Tab 1: Configuración ─────────────────────────────────────────────
    def _build_config_tab(self) -> QtWidgets.QWidget:
        w      = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner  = QtWidgets.QWidget()
        grid   = QtWidgets.QGridLayout(inner)
        grid.setSpacing(4)

        headers = [
            "Órgano", "Tipo", "N₀ (células)", "TD₅₀ (Gy)",
            "m", "n (vol.)", "α/β (Gy)", "Modelo NTCP"
        ]
        for col, h in enumerate(headers):
            lbl = QtWidgets.QLabel(f"<b>{h}</b>")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(lbl, 0, col)

        for row, organ in enumerate(self.organs, start=1):
            p = self.rb_params[organ]
            row_w: dict = {}

            # Nombre
            grid.addWidget(QtWidgets.QLabel(organ), row, 0)

            # Tipo tumor/OAR
            cb_type = QtWidgets.QComboBox()
            cb_type.addItems(["oar", "tumor"])
            cb_type.setCurrentText(p.organ_type)
            grid.addWidget(cb_type, row, 1)
            row_w["type"] = cb_type

            # N₀
            sp_N0 = QtWidgets.QDoubleSpinBox()
            sp_N0.setRange(1e3, 1e12)
            sp_N0.setDecimals(0)
            sp_N0.setSingleStep(1e6)
            sp_N0.setValue(p.N0)
            sp_N0.setEnabled(p.organ_type == "tumor")
            grid.addWidget(sp_N0, row, 2)
            row_w["N0"] = sp_N0

            # TD50
            sp_td50 = QtWidgets.QDoubleSpinBox()
            sp_td50.setRange(1.0, 300.0)
            sp_td50.setDecimals(1)
            sp_td50.setValue(p.TD50)
            sp_td50.setEnabled(p.organ_type == "oar")
            grid.addWidget(sp_td50, row, 3)
            row_w["TD50"] = sp_td50

            # m
            sp_m = QtWidgets.QDoubleSpinBox()
            sp_m.setRange(0.01, 1.0)
            sp_m.setDecimals(3)
            sp_m.setValue(p.m)
            sp_m.setEnabled(p.organ_type == "oar")
            grid.addWidget(sp_m, row, 4)
            row_w["m"] = sp_m

            # n
            sp_n = QtWidgets.QDoubleSpinBox()
            sp_n.setRange(0.001, 1.5)
            sp_n.setDecimals(3)
            sp_n.setValue(p.n)
            sp_n.setEnabled(p.organ_type == "oar")
            grid.addWidget(sp_n, row, 5)
            row_w["n"] = sp_n

            # α/β
            sp_ab = QtWidgets.QDoubleSpinBox()
            sp_ab.setRange(0.1, 50.0)
            sp_ab.setDecimals(1)
            sp_ab.setValue(p.alpha_beta)
            grid.addWidget(sp_ab, row, 6)
            row_w["ab"] = sp_ab

            # Modelo NTCP
            cb_model = QtWidgets.QComboBox()
            cb_model.addItems(_NTCP_MODEL_OPTIONS)
            cb_model.setCurrentText(p.ntcp_model)
            cb_model.setEnabled(p.organ_type == "oar")
            grid.addWidget(cb_model, row, 7)
            row_w["ntcp_model"] = cb_model

            # Habilitar/deshabilitar campos según tipo
            def _on_type_change(text, rw=row_w):
                is_t = (text == "tumor")
                rw["N0"].setEnabled(is_t)
                rw["TD50"].setEnabled(not is_t)
                rw["m"].setEnabled(not is_t)
                rw["n"].setEnabled(not is_t)
                rw["ntcp_model"].setEnabled(not is_t)

            cb_type.currentTextChanged.connect(_on_type_change)
            self._config_rows[organ] = row_w

        inner.setLayout(grid)
        scroll.setWidget(inner)
        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(scroll)
        return w

    # ── Tab 2: Resultados TCP / NTCP ─────────────────────────────────────
    def _build_results_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        self.lbl_summary = QtWidgets.QLabel("<i>Presioná Calcular para ver resultados.</i>")
        layout.addWidget(self.lbl_summary)

        self.tbl_results = QtWidgets.QTableWidget()
        self.tbl_results.setAlternatingRowColors(True)
        self.tbl_results.verticalHeader().setVisible(False)
        cols = ["Órgano","Tipo","TCP / NTCP","± (MC)","IC 90%","gEUD (Gy)","Dmean (Gy)","Dmax (Gy)"]
        self.tbl_results.setColumnCount(len(cols))
        self.tbl_results.setHorizontalHeaderLabels(cols)
        layout.addWidget(self.tbl_results)

        # Curvas TCP/NTCP vs. escala de dosis
        self.canvas_tcpntcp = MplCanvas(w, width=7, height=3)
        layout.addWidget(self.canvas_tcpntcp)
        return w

    # ── Tab 3: Supervivencia ─────────────────────────────────────────────
    def _build_survival_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        lbl = QtWidgets.QLabel("DVH de supervivencia celular S(A) por órgano.")
        layout.addWidget(lbl)
        self.canvas_surv = MplCanvas(w, width=7, height=4)
        layout.addWidget(self.canvas_surv)
        return w

    # ── Tab 4: BED / EQD2 ────────────────────────────────────────────────
    def _build_bed_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        self.tbl_bed = QtWidgets.QTableWidget()
        self.tbl_bed.setAlternatingRowColors(True)
        self.tbl_bed.verticalHeader().setVisible(False)
        cols = ["Órgano","α/β (Gy)","BED media (Gy)","BED D50 (Gy)","EQD2 media (Gy)","EQD2 D50 (Gy)"]
        self.tbl_bed.setColumnCount(len(cols))
        self.tbl_bed.setHorizontalHeaderLabels(cols)
        layout.addWidget(self.tbl_bed)
        return w

    # ─── Cálculo ──────────────────────────────────────────────────────────
    def _collect_params(self) -> dict[str, RadiobioOrganParams]:
        """Lee el estado actual de los widgets de configuración."""
        out = {}
        for organ, row_w in self._config_rows.items():
            organ_type = row_w["type"].currentText()
            out[organ] = RadiobioOrganParams(
                organ_type  = organ_type,
                N0          = float(row_w["N0"].value()),
                TD50        = float(row_w["TD50"].value()),
                m           = float(row_w["m"].value()),
                n           = float(row_w["n"].value()),
                alpha_beta  = float(row_w["ab"].value()),
                ntcp_model  = row_w["ntcp_model"].currentText(),
                run_mc      = True,
            )
        return out

    def _run_calculation(self):
        """Ejecuta compute_radiobio_report y actualiza todas las pestañas."""
        params = self._collect_params()

        try:
            self.radiobio_report = compute_radiobio_report(
                isoe_report          = self.report,
                params_by_organ      = params,
                isoe_params_by_organ = self.isoe_params_by_organ,
                N_mc_samples         = 500,
                run_mc               = True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error en cálculo", str(exc))
            return

        self._update_results_tab()
        self._update_survival_tab()
        self._update_bed_tab()
        self.tabs.setCurrentIndex(1)

    # ─── Actualización de pestañas ────────────────────────────────────────
    def _update_results_tab(self):
        rep  = self.radiobio_report
        summ = rep.get("summary", {})
        tcp_t = summ.get("TCP_total")
        max_n = summ.get("max_NTCP")

        parts = []
        if tcp_t is not None:
            parts.append(f"<b>TCP total:</b> {tcp_t:.1%}")
        if max_n is not None:
            parts.append(f"<b>NTCP máximo:</b> {max_n:.1%}")
        self.lbl_summary.setText("  &nbsp;&nbsp;  ".join(parts) or "Sin resultados.")

        organs_data = rep.get("organs", {})
        rows = list(organs_data.items())
        tbl  = self.tbl_results
        tbl.setRowCount(len(rows))

        def _it(txt):
            item = QtWidgets.QTableWidgetItem(str(txt))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            return item

        ax = self.canvas_tcpntcp.ax
        ax.clear()

        for r, (organ, data) in enumerate(rows):
            otype = data.get("organ_type","oar")
            tbl.setItem(r, 0, _it(organ))
            tbl.setItem(r, 1, _it(otype))

            mc    = data.get("mc", {})
            d_mean = 0.0
            d_max  = 0.0

            if otype == "tumor" and "tcp" in data:
                t    = data["tcp"]
                val  = t["TCP"]
                std  = mc.get("TCP_std", 0.0)
                p5   = mc.get("TCP_p5",  val)
                p95  = mc.get("TCP_p95", val)
                d_mean = t.get("D_mean_Gy", 0.0)
                tbl.setItem(r, 2, _it(f"{val:.1%}"))
                tbl.setItem(r, 3, _it(f"±{std:.1%}" if std else "—"))
                tbl.setItem(r, 4, _it(f"[{p5:.1%}, {p95:.1%}]" if std else "—"))
                tbl.setItem(r, 5, _it("—"))
                # Curva TCP
                scales = np.linspace(0, 2, 80)
                iso_p  = self.isoe_params_by_organ.get(organ)
                p_rb   = self._collect_params().get(organ)
                if iso_p and p_rb:
                    A = np.asarray(self.report["IsoVoxel"].get(organ, []), float)
                    from ratmaster.physics.radiobio.tcp import tcp_curve
                    curve = tcp_curve(scales, A, iso_p.aR, iso_p.bR, iso_p.GR, p_rb.N0)
                    ax.plot(scales, curve * 100, label=f"TCP {organ}", linestyle="--")

            elif "ntcp" in data:
                n    = data["ntcp"]
                val  = n["NTCP"]
                std  = mc.get("NTCP_std", 0.0)
                p5   = mc.get("NTCP_p5",  val)
                p95  = mc.get("NTCP_p95", val)
                d_mean = n.get("D_mean_Gy", 0.0)
                d_max  = n.get("D_max_Gy",  0.0)
                geud_v = n.get("gEUD_Gy",   0.0)
                tbl.setItem(r, 2, _it(f"{val:.1%}"))
                tbl.setItem(r, 3, _it(f"±{std:.1%}" if std else "—"))
                tbl.setItem(r, 4, _it(f"[{p5:.1%}, {p95:.1%}]" if std else "—"))
                tbl.setItem(r, 5, _it(f"{geud_v:.2f}"))

            tbl.setItem(r, 6, _it(f"{d_mean:.2f}"))
            tbl.setItem(r, 7, _it(f"{d_max:.2f}"))

        tbl.resizeColumnsToContents()

        ax.set_xlabel("Factor de escala de dosis")
        ax.set_ylabel("Probabilidad (%)")
        ax.set_title("Curvas TCP / NTCP vs. escala de dosis")
        ax.set_xlim(0, 2); ax.set_ylim(0, 105)
        ax.axvline(1.0, color="gray", linestyle=":", lw=1, label="Dosis calculada")
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.4)
        self.canvas_tcpntcp.fig.tight_layout(pad=1.2)
        self.canvas_tcpntcp.draw()

    def _update_survival_tab(self):
        ax = self.canvas_surv.ax
        ax.clear()
        iso_vox = self.report.get("IsoVoxel", {})
        colors  = _ORGAN_COLORS

        for idx, (organ, A_list) in enumerate(iso_vox.items()):
            iso_p = self.isoe_params_by_organ.get(organ)
            if iso_p is None:
                continue
            A = np.asarray(A_list, float)
            if A.size == 0:
                continue
            S = survival_voxel(A, iso_p.aR, iso_p.bR, iso_p.GR)
            # DVH de supervivencia: S en el eje X, volumen en el eje Y
            s_sorted, vol = build_dvh(S)
            s_sorted, vol = dvh_extend_to_zero(s_sorted, vol)
            if s_sorted.size > 0:
                ax.plot(s_sorted, vol, label=organ,
                        color=colors[idx % len(colors)])

        ax.set_xlabel("Supervivencia celular S(A)")
        ax.set_ylabel("Volumen (%)")
        ax.set_title("DVH de supervivencia — S(A) = exp(−αA − Gβ A²)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, framealpha=0.85)
        self.canvas_surv.fig.tight_layout(pad=1.2)
        self.canvas_surv.draw()

    def _update_bed_tab(self):
        organs_data = self.radiobio_report.get("organs", {})
        tbl = self.tbl_bed
        rows = [(k, v) for k, v in organs_data.items() if "bed_eqd2" in v]
        tbl.setRowCount(len(rows))

        def _it(txt):
            item = QtWidgets.QTableWidgetItem(str(txt))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            return item

        for r, (organ, data) in enumerate(rows):
            b = data["bed_eqd2"]
            tbl.setItem(r, 0, _it(organ))
            tbl.setItem(r, 1, _it(f"{b['alpha_beta_used']:.1f}"))
            tbl.setItem(r, 2, _it(f"{b['BED_mean']:.2f}"))
            tbl.setItem(r, 3, _it(f"{b['BED_D50']:.2f}"))
            tbl.setItem(r, 4, _it(f"{b['EQD2_mean']:.2f}"))
            tbl.setItem(r, 5, _it(f"{b['EQD2_D50']:.2f}"))
        tbl.resizeColumnsToContents()
