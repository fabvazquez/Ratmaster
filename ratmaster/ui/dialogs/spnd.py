"""
ui/dialogs/spnd.py
==================
Diálogos de configuración y cálculo del detector SPND.

SPNDConfigDialog: gestor del registro de detectores (sensibilidad, factores).
SPNDFromCurrentsDialog: asistente para calcular flujo neutrónico
    a partir de corrientes medidas del SPND.
"""

import json
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.data.persistence import (
    load_spnd_registry, save_spnd_registry, parse_number_or_pair,
)
from ratmaster.ui.formatters import format_scientific_value_uncertainty
from ratmaster.ui.dialogs.constraints import PasteableTable


class SPNDConfigDialog(QtWidgets.QDialog):
    """Editar/guardar la libreta de detectores (factor, sensibilidad, fecha)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SPND — Configuración y calibración")
        self.resize(900, 420)
        self.reg = load_spnd_registry()

        lay = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.lbl_updated = QtWidgets.QLabel(f"<b>{self.reg.get('updated_label','')}</b>")
        top.addWidget(self.lbl_updated)
        top.addStretch()
        btn_add = QtWidgets.QPushButton("Agregar detector")
        btn_del = QtWidgets.QPushButton("Eliminar seleccionado")
        top.addWidget(btn_add); top.addWidget(btn_del)
        lay.addLayout(top)

        self.tbl = PasteableTable()
        self.tbl.setColumnCount(6)
        self.tbl.setHorizontalHeaderLabels([
            "Nombre", "Factor a Verde", "Sens (A/nv)", "σ Sens", "Últ. calib (YYYY-MM-DD)", "Notas"
        ])
        self.tbl.setRowCount(0)
        lay.addWidget(self.tbl)

        # cargar
        for d in self.reg.get("detectors", []):
            self._append_row(d)

        self.tbl.resizeColumnsToContents()

        btns = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Guardar")
        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        btns.addStretch(); btns.addWidget(self.btn_save); btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)

        btn_add.clicked.connect(self._add)
        btn_del.clicked.connect(self._del)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._save)

    def _append_row(self, d=None):
        d = d or {}
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        def setc(col, val):
            it = QtWidgets.QTableWidgetItem("" if val is None else str(val))
            self.tbl.setItem(r, col, it)
        setc(0, d.get("name",""))
        setc(1, d.get("factor_to_verde", ""))
        setc(2, d.get("sens", ""))
        setc(3, d.get("sens_sigma", ""))
        setc(4, d.get("last_calib", ""))
        setc(5, d.get("notes", ""))

    def _add(self):
        self._append_row({"name":"", "factor_to_verde":1.0, "sens":"", "sens_sigma":"", "last_calib":"", "notes":""})
        self.tbl.scrollToBottom()

    def _del(self):
        r = self.tbl.currentRow()
        if r >= 0:
            self.tbl.removeRow(r)

    def _save(self):
        dets = []
        for r in range(self.tbl.rowCount()):
            name = (self.tbl.item(r,0).text() if self.tbl.item(r,0) else "").strip()
            if not name:
                continue
            f_txt = self.tbl.item(r,1).text() if self.tbl.item(r,1) else ""
            f_val, _ = parse_number_or_pair(f_txt)
            if f_val is None:
                f_val = 1.0

            s_txt = self.tbl.item(r,2).text() if self.tbl.item(r,2) else ""
            s_val, _ = parse_number_or_pair(s_txt)
            ss_txt = self.tbl.item(r,3).text() if self.tbl.item(r,3) else ""
            ss_val, _ = parse_number_or_pair(ss_txt)

            last_calib = (self.tbl.item(r,4).text() if self.tbl.item(r,4) else "").strip()
            notes = (self.tbl.item(r,5).text() if self.tbl.item(r,5) else "").strip()

            dets.append({
                "name": name,
                "factor_to_verde": float(f_val),
                "sens": None if s_val is None else float(s_val),
                "sens_sigma": None if ss_val is None else float(ss_val),
                "last_calib": last_calib,
                "notes": notes
            })

        self.reg["detectors"] = dets
        ok, msg = save_spnd_registry(self.reg)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "SPND", "No pude guardar spnd_registry.json:\n"+msg)
            return
        self.accept()

def _det_by_name(reg, name):
    for d in reg.get("detectors", []):
        if d.get("name") == name:
            return d
    return None



class SPNDFromCurrentsDialog(QtWidgets.QDialog):
    """Asistente: corrientes -> flujo."""
    def __init__(self, parent=None, snapshot=None):
        super().__init__(parent)
        self.setWindowTitle("SPND — Calcular flujo desde corrientes")
        self.resize(1020, 620)
        self.reg = load_spnd_registry()

        self._phi = None
        self._sigphi = None
        self._snapshot = None   # se llena en _recalc y al restaurar

        lay = QtWidgets.QVBoxLayout(self)

        # =========================================================
        # Header: referencia + botones
        # =========================================================
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Detector de referencia:"))

        self.cmb_ref = QtWidgets.QComboBox()
        det_names = [d["name"] for d in self.reg.get("detectors", [])]
        self.cmb_ref.addItems(det_names)

        if "Verde (ref)" in det_names:
            self.cmb_ref.setCurrentText("Verde (ref)")
        elif det_names:
            self.cmb_ref.setCurrentIndex(0)

        top.addWidget(self.cmb_ref)

        self.lbl_updated = QtWidgets.QLabel(self.reg.get("updated_label", ""))
        self.lbl_updated.setStyleSheet("color: #555;")
        top.addSpacing(12)
        top.addWidget(self.lbl_updated)

        top.addStretch()

        btn_cfg = QtWidgets.QPushButton("Configurar/calibración…")
        top.addWidget(btn_cfg)
        lay.addLayout(top)

        # =========================================================
        # Tabla
        # =========================================================
        self.tbl = PasteableTable()
        self.tbl.setColumnCount(8)
        self.tbl.setHorizontalHeaderLabels([
            "Usar",
            "Detector",
            "I medida (pA)",
            "± I (pA)",
            "Factor → ref",
            "I ref equiv (pA)",
            "± I ref (pA)",
            "Últ. calib."
        ])
        # Estilo del indicador de checkbox igual al de org_list en la ventana principal:
        # sin selección de fila, el color se aplica solo dentro del cuadrado del checkbox.
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tbl.setStyleSheet("""
QTableWidget {
    gridline-color: #CFD8DC;
}
QTableWidget::indicator {
    width: 14px;
    height: 14px;
    border: 2px solid #90A4AE;
    border-radius: 3px;
    background-color: #FFFFFF;
}
QTableWidget::indicator:checked {
    background-color: #37474F;
    border-color: #263238;
}
""")
        self.tbl.setRowCount(3)
        self._init_row(0, "Rojo")
        self._init_row(1, "A0")
        self._init_row(2, "A1")
        lay.addWidget(self.tbl)
        self.tbl.resizeColumnsToContents()

        # =========================================================
        # CIP
        # =========================================================
        cip = QtWidgets.QGroupBox("Reajuste por variación de CIP")
        f = QtWidgets.QGridLayout(cip)

        self.in_cip_mon = QtWidgets.QLineEdit("")
        self.in_cip_irr = QtWidgets.QLineEdit("")
        self.in_cip_pert = QtWidgets.QLineEdit("")

        self.out_pedir_cippert = self._make_value_label()

        f.addWidget(QtWidgets.QLabel("CIP en monitoreo"), 0, 0)
        f.addWidget(self.in_cip_mon, 0, 1)

        f.addWidget(QtWidgets.QLabel("CIP en irradiación"), 0, 2)
        f.addWidget(self.in_cip_irr, 0, 3)

        f.addWidget(QtWidgets.QLabel("CIP pert"), 1, 0)
        f.addWidget(self.in_cip_pert, 1, 1)

        f.addWidget(QtWidgets.QLabel("Pedir CIP pert"), 1, 2)
        f.addWidget(self.out_pedir_cippert, 1, 3)

        lay.addWidget(cip)

        # =========================================================
        # Resultados
        # =========================================================
        res = QtWidgets.QGroupBox("Resultados")
        g = QtWidgets.QGridLayout(res)

        self.out_Imean = self._make_value_label()
        self.out_Iirr = self._make_value_label()
        self.out_phi = self._make_value_label()
        self.out_ref = self._make_value_label()
        self.out_sens = self._make_value_label()
        self.out_break = self._make_value_label()

        g.addWidget(QtWidgets.QLabel("Corriente media equivalente"), 0, 0)
        g.addWidget(self.out_Imean, 0, 1)

        g.addWidget(QtWidgets.QLabel("Corriente SPND en irradiación"), 1, 0)
        g.addWidget(self.out_Iirr, 1, 1)

        g.addWidget(QtWidgets.QLabel("Flujo térmico Φ"), 2, 0)
        g.addWidget(self.out_phi, 2, 1)

        g.addWidget(QtWidgets.QLabel("Detector de referencia usado"), 3, 0)
        g.addWidget(self.out_ref, 3, 1)

        g.addWidget(QtWidgets.QLabel("Sensibilidad usada"), 4, 0)
        g.addWidget(self.out_sens, 4, 1)

        g.addWidget(QtWidgets.QLabel("Detalle incertidumbre"), 5, 0)
        g.addWidget(self.out_break, 5, 1)

        lay.addWidget(res)

        # =========================================================
        # Botones
        # =========================================================
        btns = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("Calcular")
        self.btn_apply = QtWidgets.QPushButton("Aplicar a RatMaster")
        self.btn_close = QtWidgets.QPushButton("Cerrar")

        btns.addWidget(self.btn_calc)
        btns.addWidget(self.btn_apply)
        btns.addStretch()
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.btn_apply.setEnabled(False)

        # =========================================================
        # Signals
        # =========================================================
        self.btn_close.clicked.connect(self.reject)
        self.btn_calc.clicked.connect(self._recalc)
        self.btn_apply.clicked.connect(self.accept)
        btn_cfg.clicked.connect(self._open_cfg)

        self.cmb_ref.currentTextChanged.connect(lambda _: self._refresh_factors())
        self.tbl.itemChanged.connect(lambda *_: self._refresh_factors())
        self.tbl.itemChanged.connect(lambda *_: self._update_spnd_row_colors())
        self.in_cip_mon.textChanged.connect(lambda *_: self._update_cip_preview())
        self.in_cip_irr.textChanged.connect(lambda *_: self._update_cip_preview())
        self.in_cip_pert.textChanged.connect(lambda *_: self._update_cip_preview())

        # refresco final cuando ya existen todos los widgets
        self._refresh_factors()
        self._update_cip_preview()
        self._update_spnd_row_colors()

        # Si hay un snapshot previo, restaurar corrientes y CIP para que el usuario
        # vea exactamente los valores que usó la última vez.
        if snapshot:
            self._restore_snapshot(snapshot)

    # ── Color de cuadradito checkbox según estado "Usar" ─────────────────
    def _update_spnd_row_colors(self):
        """
        El color del indicador del checkbox se maneja via stylesheet en self.tbl
        (QTableWidget::indicator:checked / :unchecked), igual que en org_list.
        Esta función solo fuerza un repintado cuando el estado de checked cambia.
        """
        self.tbl.viewport().update()
    # =========================================================
    # Helpers UI
    # =========================================================
    def _make_value_label(self):
        lbl = QtWidgets.QLabel("—")
        lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lbl.setStyleSheet("""
            QLabel {
                background: #f7f7f7;
                border: 1px solid #d8d8d8;
                border-radius: 6px;
                padding: 6px 8px;
            }
        """)
        lbl.setWordWrap(True)
        return lbl

    # =========================================================
    # Helpers numéricos / formato
    # =========================================================
    def _format_uncertainty(self, value, sigma, unit=""):
        """
        Formato tipo:
        12.34 ± 0.56 pA
        usando cifras significativas razonables para sigma.
        """
        if value is None:
            return "—"

        if sigma is None:
            return f"{float(value):.6g}" + (f" {unit}" if unit else "")

        value = float(value)
        sigma = abs(float(sigma))

        if sigma == 0:
            txt = f"{value:.6g} ± 0"
            return txt + (f" {unit}" if unit else "")

        import math

        exp_sigma = math.floor(math.log10(sigma))
        mant_sigma = sigma / (10 ** exp_sigma)

        # 2 cifras si empieza en 1 o 2; si no 1 cifra
        sig_digits = 2 if mant_sigma < 3 else 1
        decimals = -exp_sigma + (sig_digits - 1)

        sigma_r = round(sigma, decimals)
        value_r = round(value, decimals)

        if decimals > 0:
            fmt = "{:." + str(decimals) + "f}"
            txt = f"{fmt.format(value_r)} ± {fmt.format(sigma_r)}"
        else:
            txt = f"{round(value_r):.0f} ± {round(sigma_r):.0f}"

        if unit:
            txt += f" {unit}"
        return txt

    def _format_scientific_uncertainty(self, value, sigma, unit=""):
        """
        Formato tipo:
        (3.21 ± 0.18) × 10^8 n/cm²·s
        """
        if value is None:
            return "—"

        if sigma is None:
            return f"{float(value):.6g}" + (f" {unit}" if unit else "")

        value = float(value)
        sigma = abs(float(sigma))

        if value == 0:
            return self._format_uncertainty(value, sigma, unit)

        import math

        expv = int(math.floor(math.log10(abs(value))))
        scale = 10 ** expv

        v_scaled = value / scale
        s_scaled = sigma / scale

        core = self._format_uncertainty(v_scaled, s_scaled, "")
        txt = f"({core}) × 10^{expv}"
        if unit:
            txt += f" {unit}"
        return txt

    # =========================================================
    # Inicialización de filas
    # =========================================================
    def _init_row(self, r, det_name):
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
        chk.setCheckState(
            QtCore.Qt.Checked if det_name in ("Rojo", "A1") else QtCore.Qt.Unchecked
        )
        self.tbl.setItem(r, 0, chk)

        cmb = QtWidgets.QComboBox()
        det_names = [d["name"] for d in self.reg.get("detectors", [])]
        cmb.addItems(det_names)

        if det_name in det_names:
            cmb.setCurrentText(det_name)
        elif det_names:
            cmb.setCurrentIndex(0)

        self.tbl.setCellWidget(r, 1, cmb)
        cmb.currentTextChanged.connect(lambda *_: self._row_update(r))

        self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(""))
        self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(""))

        for c in (4, 5, 6, 7):
            it = QtWidgets.QTableWidgetItem("—")
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, c, it)

    # =========================================================
    # Config
    # =========================================================
    def _open_cfg(self):
        dlg = SPNDConfigDialog(self)
        if dlg.exec():
            self.reg = load_spnd_registry()

            det_names = [d["name"] for d in self.reg.get("detectors", [])]

            cur = self.cmb_ref.currentText()
            self.cmb_ref.blockSignals(True)
            self.cmb_ref.clear()
            self.cmb_ref.addItems(det_names)
            if cur in det_names:
                self.cmb_ref.setCurrentText(cur)
            elif det_names:
                self.cmb_ref.setCurrentIndex(0)
            self.cmb_ref.blockSignals(False)

            for r in range(self.tbl.rowCount()):
                w = self.tbl.cellWidget(r, 1)
                if isinstance(w, QtWidgets.QComboBox):
                    curw = w.currentText()
                    w.blockSignals(True)
                    w.clear()
                    w.addItems(det_names)
                    if curw in det_names:
                        w.setCurrentText(curw)
                    elif det_names:
                        w.setCurrentIndex(0)
                    w.blockSignals(False)

            self.lbl_updated.setText(self.reg.get("updated_label", ""))
            self._refresh_factors()

    # =========================================================
    # Factores
    # =========================================================
    def _factor_det_to_ref(self, det_name, ref_name):
        det = _det_by_name(self.reg, det_name)
        ref = _det_by_name(self.reg, ref_name)

        if not det or not ref:
            return None

        fdv = float(det.get("factor_to_verde", 1.0) or 1.0)
        frv = float(ref.get("factor_to_verde", 1.0) or 1.0)

        if frv == 0:
            return None

        return fdv / frv

    def _row_update(self, r):
        self._refresh_factors()

    def _refresh_factors(self):
        ref = self.cmb_ref.currentText()

        for r in range(self.tbl.rowCount()):
            w = self.tbl.cellWidget(r, 1)
            det = w.currentText() if isinstance(w, QtWidgets.QComboBox) else ""

            f = self._factor_det_to_ref(det, ref)
            last = (_det_by_name(self.reg, det) or {}).get("last_calib", "")

            def _ensure(col):
                it = self.tbl.item(r, col)
                if it is None:
                    it = QtWidgets.QTableWidgetItem("—")
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.tbl.setItem(r, col, it)
                return it

            _ensure(4).setText("—" if f is None else f"{f:.9g}")
            _ensure(7).setText(last if last else "—")

            I, sig_pair = parse_number_or_pair(
                self.tbl.item(r, 2).text() if self.tbl.item(r, 2) else ""
            )

            sig = None
            s3, _ = parse_number_or_pair(
                self.tbl.item(r, 3).text() if self.tbl.item(r, 3) else ""
            )
            if s3 is not None:
                sig = abs(s3)
            elif sig_pair is not None:
                sig = abs(sig_pair)

            if I is None or f is None:
                _ensure(5).setText("—")
                _ensure(6).setText("—")
            else:
                Ieq = I * f
                _ensure(5).setText(f"{Ieq:.9g}")
                if sig is None:
                    _ensure(6).setText("—")
                else:
                    _ensure(6).setText(f"{abs(sig) * abs(f):.9g}")

        ref_name = self.cmb_ref.currentText()
        ref_det = _det_by_name(self.reg, ref_name) or {}
        S = ref_det.get("sens", None)
        Ssig = ref_det.get("sens_sigma", None)

        if hasattr(self, "out_ref"):
            self.out_ref.setText(ref_name if ref_name else "—")

        if hasattr(self, "out_sens"):
            if S is None or Ssig is None:
                self.out_sens.setText("— (faltan S y/o σ(S))")
            else:
                self.out_sens.setText(
                    self._format_scientific_uncertainty(float(S), float(Ssig), "A/nv")
                )

        if hasattr(self, "out_Imean"):
            self.out_Imean.setText("—")
        if hasattr(self, "out_Iirr"):
            self.out_Iirr.setText("—")
        if hasattr(self, "out_phi"):
            self.out_phi.setText("—")
        if hasattr(self, "out_break"):
            self.out_break.setText("—")
        if hasattr(self, "out_pedir_cippert"):
            self.out_pedir_cippert.setText("—")

        if hasattr(self, "btn_apply"):
            self.btn_apply.setEnabled(False)

        self._phi = None
        self._sigphi = None

    # =========================================================
    # Cálculo
    # =========================================================
    def _update_cip_preview(self):
        cip_mon, _ = parse_number_or_pair(self.in_cip_mon.text())
        cip_irr, _ = parse_number_or_pair(self.in_cip_irr.text())
        cip_pert, _ = parse_number_or_pair(self.in_cip_pert.text())

        if cip_mon is None or cip_irr is None or cip_pert is None or cip_mon == 0 or cip_pert == 0:
            self.out_pedir_cippert.setText("—")
            return

        pedir_cippert = float(cip_irr * cip_pert / cip_mon)
        self.out_pedir_cippert.setText(f"{pedir_cippert:.6g}")

    def _recalc(self):
        ref = self.cmb_ref.currentText()
        ref_det = _det_by_name(self.reg, ref) or {}

        S = ref_det.get("sens", None)
        Ssig = ref_det.get("sens_sigma", None)

        if S is None or Ssig is None:
            QtWidgets.QMessageBox.warning(
                self,
                "SPND",
                f"Falta sensibilidad y/o σ(S) para el detector de referencia: {ref}.\n"
                "Configuralo en 'Configurar/calibración…'."
            )
            return

        S = float(S)
        Ssig = float(Ssig)

        Ieq_list = []
        Sig_list = []

        for r in range(self.tbl.rowCount()):
            use_it = self.tbl.item(r, 0)
            use = (use_it.checkState() == QtCore.Qt.Checked) if use_it else False
            if not use:
                continue

            w = self.tbl.cellWidget(r, 1)
            det = w.currentText() if isinstance(w, QtWidgets.QComboBox) else ""

            f = self._factor_det_to_ref(det, ref)
            if f is None:
                continue

            I, sig_pair = parse_number_or_pair(
                self.tbl.item(r, 2).text() if self.tbl.item(r, 2) else ""
            )
            if I is None:
                continue

            s3, _ = parse_number_or_pair(
                self.tbl.item(r, 3).text() if self.tbl.item(r, 3) else ""
            )

            if s3 is not None:
                sig = abs(s3)
            else:
                sig = abs(sig_pair) if sig_pair is not None else None

            if sig is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "SPND",
                    "Falta incertidumbre (± I) en una de las filas marcadas como 'Usar'."
                )
                return

            Ieq = I * f
            Ieq_list.append(float(Ieq))
            Sig_list.append(float(abs(sig) * abs(f)))

        if len(Ieq_list) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "SPND",
                "No hay filas válidas marcadas como 'Usar'."
            )
            return

        # ---------------------------------------------------------
        # Corriente media equivalente
        # ---------------------------------------------------------
        Imean = float(np.mean(Ieq_list))

        # criterio conservador, mismo que venías usando
        sigI = float(np.sqrt(np.sum(np.array(Sig_list, dtype=float) ** 2)))

        # ---------------------------------------------------------
        # CIP
        # ---------------------------------------------------------
        cip_mon, _ = parse_number_or_pair(self.in_cip_mon.text())
        cip_irr, _ = parse_number_or_pair(self.in_cip_irr.text())
        cip_pert, _ = parse_number_or_pair(self.in_cip_pert.text())

        if cip_mon is None or cip_irr is None or cip_pert is None:
            QtWidgets.QMessageBox.warning(
                self,
                "SPND",
                "Tenés que completar CIP en monitoreo, CIP en irradiación y CIP pert."
            )
            return

        if cip_mon == 0 or cip_pert == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "SPND",
                "CIP en monitoreo y CIP pert deben ser distintos de cero."
            )
            return

        # Pedir CIP pert = (CIP irr * CIP pert / CIP mon)
        pedir_cippert = float(cip_irr * cip_pert / cip_mon)
        self._update_cip_preview()

        # Corriente ajustada a irradiación
        ratio = float(pedir_cippert / cip_pert)   # = cip_irr / cip_mon
        Iirr = float(Imean * ratio)
        sigIirr = float(sigI * abs(ratio))        # no incluye error de CIP

        # ---------------------------------------------------------
        # Flujo
        # ---------------------------------------------------------
        phi = float((Iirr * 1e-12) / S)   # I en pA -> A
        rel_I = abs(sigIirr) / max(abs(Iirr), 1e-30)
        rel_S = abs(Ssig) / max(abs(S), 1e-30)
        rel_phi = float(np.sqrt(rel_I ** 2 + rel_S ** 2))
        sigphi = float(abs(phi) * rel_phi)

        self._phi = phi
        self._sigphi = sigphi

        # ---------------------------------------------------------
        # Mostrar resultados
        # ---------------------------------------------------------
        self.out_Imean.setText(self._format_uncertainty(Imean, sigI, "pA"))
        self.out_Iirr.setText(self._format_uncertainty(Iirr, sigIirr, "pA"))
        self.out_phi.setText(
            self._format_scientific_uncertainty(phi, sigphi, "n/cm²·s")
        )
        self.out_ref.setText(ref)
        self.out_sens.setText(
            self._format_scientific_uncertainty(S, Ssig, "A/nv")
        )
        self.out_break.setText(
            f"rel(Iirr) = {rel_I*100:.3g}%   |   "
            f"rel(S) = {rel_S*100:.3g}%   |   "
            f"rel(Φ) = {rel_phi*100:.3g}%"
        )

        self.btn_apply.setEnabled(True)

    # =========================================================
    # Salida
    # =========================================================
    def get_result(self):
        """Devuelve (phi, sigma_phi) en n/cm²·s."""
        return self._phi, self._sigphi

    # ── Snapshot: persistencia entre sesiones ──────────────────────────

    def _restore_snapshot(self, snap: dict):
        """Repopula la tabla y CIPs desde un snapshot guardado."""
        # Detector de referencia
        ref = snap.get("ref_detector", "")
        if ref:
            idx = self.cmb_ref.findText(ref)
            if idx >= 0:
                self.cmb_ref.setCurrentIndex(idx)
        # Filas
        rows = snap.get("rows", [])
        for r, row_data in enumerate(rows):
            if r >= self.tbl.rowCount():
                break
            # Detector
            w = self.tbl.cellWidget(r, 1)
            if isinstance(w, QtWidgets.QComboBox):
                det_name = row_data.get("detector", "")
                idx = w.findText(det_name)
                if idx >= 0:
                    w.setCurrentIndex(idx)
            # Checkbox usar
            chk = self.tbl.item(r, 0)
            if chk:
                chk.setCheckState(
                    QtCore.Qt.Checked if row_data.get("usar", False) else QtCore.Qt.Unchecked
                )
            # Corriente medida y sigma
            it2 = self.tbl.item(r, 2)
            if it2:
                it2.setText(row_data.get("I_meas", ""))
            it3 = self.tbl.item(r, 3)
            if it3:
                it3.setText(row_data.get("I_sigma", ""))
        # CIPs
        self.in_cip_mon.setText(snap.get("cip_mon", ""))
        self.in_cip_irr.setText(snap.get("cip_irr", ""))
        self.in_cip_pert.setText(snap.get("cip_pert", ""))
        # Resultados ya calculados
        phi = snap.get("phi")
        sigphi = snap.get("sigphi")
        if phi is not None and sigphi is not None:
            self._phi = phi
            self._sigphi = sigphi
            self.btn_apply.setEnabled(True)
            # Repoblar labels de resultado
            for attr, key in [("out_phi", "phi_txt"), ("out_Imean", "Imean_txt"),
                            ("out_Iirr", "Iirr_txt"), ("out_sens", "sens_txt"),
                            ("out_ref", "ref_txt"), ("out_break", "break_txt")]:
                val = snap.get(key, "")
                if val and hasattr(self, attr):
                    getattr(self, attr).setText(val)
        self._refresh_factors()
        self._update_spnd_row_colors()

    def get_snapshot(self) -> dict:
        """Devuelve un dict con todo lo necesario para restaurar el estado y generar el PDF."""
        rows = []
        for r in range(self.tbl.rowCount()):
            chk = self.tbl.item(r, 0)
            w = self.tbl.cellWidget(r, 1)
            it2 = self.tbl.item(r, 2)
            it3 = self.tbl.item(r, 3)
            rows.append({
                "usar": chk.checkState() == QtCore.Qt.Checked if chk else False,
                "detector": w.currentText() if isinstance(w, QtWidgets.QComboBox) else "",
                "I_meas": it2.text() if it2 else "",
                "I_sigma": it3.text() if it3 else "",
            })
        # Capturar texto de los labels de resultado para restaurarlos visualmente
        def _lbl(attr): return getattr(self, attr).text() if hasattr(self, attr) else ""
        return {
            "ref_detector": self.cmb_ref.currentText(),
            "rows": rows,
            "cip_mon": self.in_cip_mon.text(),
            "cip_irr": self.in_cip_irr.text(),
            "cip_pert": self.in_cip_pert.text(),
            "phi": self._phi,
            "sigphi": self._sigphi,
            "phi_txt":   _lbl("out_phi"),
            "Imean_txt": _lbl("out_Imean"),
            "Iirr_txt":  _lbl("out_Iirr"),
            "sens_txt":  _lbl("out_sens"),
            "ref_txt":   _lbl("out_ref"),
            "break_txt": _lbl("out_break"),
        }
# =================== Generación de vectores desde SEG + Meshtal (UI) ===================

