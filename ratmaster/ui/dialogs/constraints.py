"""
ui/dialogs/constraints.py
=========================
Diálogo de constraints dosimétricas por órgano.

Permite definir Dmax, Dmean, Dmin y Dose@Vx% para cada órgano.
Incluye PasteableTable para pegar grillas desde Excel.
"""

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.constants import (
    ORG_ORDER, CONSTRAINT_PRESETS, USER_CONSTRAINT_PRESETS,
    _make_constraints_matrix, _is_builtin_constraint_preset,
)
from ratmaster.data.persistence import parse_number_or_pair


class PasteableTable(QtWidgets.QTableWidget):
    """QTableWidget que permite pegar grillas TSV desde Excel."""
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.matches(QtGui.QKeySequence.Paste):
            self._paste_from_clipboard()
            return
        super().keyPressEvent(e)

    def _paste_from_clipboard(self):
        cb = QtWidgets.QApplication.clipboard()
        txt = cb.text()
        if not txt:
            return
        rows = [r for r in txt.splitlines() if r.strip() != ""]
        if not rows:
            return
        data = [r.split("\t") for r in rows]

        r0 = self.currentRow()
        c0 = self.currentColumn()
        if r0 < 0: r0 = 0
        if c0 < 0: c0 = 0

        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                r = r0 + i
                c = c0 + j
                if r >= self.rowCount() or c >= self.columnCount():
                    continue
                it = self.item(r, c)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    self.setItem(r, c, it)
                it.setText(cell)




class ConstraintsDialog(QtWidgets.QDialog):
    """
    Popup para editar constraints con presets o modo manual.
    """
    def __init__(self, parent, tbl_cons: QtWidgets.QTableWidget, preset_name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Restriciones de Dosis (editar)")
        self.resize(980, 420)
        self._norg = len(ORG_ORDER)
        self._selected_name = preset_name.strip() if preset_name else ""
        self._loading = False

        lay = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Preset:"))
        self.cmb_preset = QtWidgets.QComboBox()
        self._reload_combo()
        top.addWidget(self.cmb_preset, 1)
        self.btn_save = QtWidgets.QPushButton("Guardar como preset…")
        self.btn_delete = QtWidgets.QPushButton("Borrar preset")
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_delete)
        lay.addLayout(top)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color:#546E7A; font-size:9pt;")
        lay.addWidget(self.lbl_info)

        self.tbl = QtWidgets.QTableWidget(5, self._norg)
        self.tbl.setVerticalHeaderLabels(["Dosis máxima","Dosis media","Dosis mínima","% del volumen más caliente","Dosis en ese % (Dx)"])
        self.tbl.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        lay.addWidget(self.tbl, 1)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.btn_save.clicked.connect(self._save_as_preset)
        self.btn_delete.clicked.connect(self._delete_current_preset)
        self.cmb_preset.currentTextChanged.connect(self._on_combo_changed)
        self.tbl.itemChanged.connect(self._on_table_edited)

        if preset_name and preset_name in CONSTRAINT_PRESETS:
            blocker = QtCore.QSignalBlocker(self.cmb_preset)
            self.cmb_preset.setCurrentText(preset_name)
            del blocker
            self._selected_name = preset_name
            self._set_matrix(CONSTRAINT_PRESETS[preset_name])
            self.lbl_info.setText(f"Preset seleccionado: {preset_name}")
        else:
            blocker = QtCore.QSignalBlocker(self.cmb_preset)
            self.cmb_preset.setCurrentText("Elegir manualmente")
            del blocker
            self._copy_from(tbl_cons)
            self._selected_name = ""
            self.lbl_info.setText("Restricciones de Dosis manuales.")

    def _reload_combo(self):
        self.cmb_preset.clear()
        self.cmb_preset.addItem("Elegir manualmente")
        for name in sorted(CONSTRAINT_PRESETS.keys()):
            self.cmb_preset.addItem(name)

    def _copy_from(self, tbl_cons):
        self._loading = True
        try:
            for i in range(self.tbl.rowCount()):
                for j in range(self.tbl.columnCount()):
                    it = tbl_cons.item(i, j)
                    txt = it.text().strip() if it else "0"
                    self.tbl.setItem(i, j, QtWidgets.QTableWidgetItem(txt if txt else "0"))
        finally:
            self._loading = False

    def _set_matrix(self, mat):
        self._loading = True
        try:
            for i in range(5):
                for j in range(self._norg):
                    self.tbl.setItem(i, j, QtWidgets.QTableWidgetItem(f"{float(mat[i,j]):.6g}"))
        finally:
            self._loading = False

    def _on_combo_changed(self, text):
        text = (text or "").strip()
        if text == "Elegir manualmente":
            self._selected_name = ""
            self._set_matrix(_make_constraints_matrix(0.0))
            self.lbl_info.setText("Restricción de Dosis manuales.")
            return
        mat = CONSTRAINT_PRESETS.get(text)
        if mat is None:
            return
        self._set_matrix(mat)
        self._selected_name = text
        self.lbl_info.setText(f"Preset seleccionado: {text}")

    def _on_table_edited(self, item=None):
        if self._loading:
            return
        blocker = QtCore.QSignalBlocker(self.cmb_preset)
        self.cmb_preset.setCurrentText("Elegir manualmente")
        del blocker
        self._selected_name = ""
        self.lbl_info.setText("Restricciones de Dosis manuales.")

    def _read_matrix(self):
        cons = np.zeros((5, self._norg), dtype=float)
        for i in range(5):
            for j in range(self._norg):
                it = self.tbl.item(i, j)
                txt = (it.text() if it else "0").strip().replace(",", ".")
                val = float(txt) if txt else 0.0
                if val < 0:
                    raise ValueError(f"No se permiten valores negativos ({ORG_ORDER[j]}).")
                if i == 3 and val > 100:
                    raise ValueError(f"El porcentaje (%) no puede ser mayor que 100 ({ORG_ORDER[j]}).")
                cons[i, j] = val
        return cons

    def _save_as_preset(self):
        try:
            mat = self._read_matrix()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Restriciones de Dosis", str(e))
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Guardar preset", "Nombre del nuevo preset:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in CONSTRAINT_PRESETS:
            if QtWidgets.QMessageBox.question(self, "Sobrescribir", f"Ya existe '{name}'. ¿Sobrescribirlo?") != QtWidgets.QMessageBox.Yes:
                return
        CONSTRAINT_PRESETS[name] = mat.copy()
        self._selected_name = name
        self._reload_combo()
        self.cmb_preset.setCurrentText(name)
        self.lbl_info.setText(f"Preset guardado: {name}")

    def _delete_current_preset(self):
        name = self.cmb_preset.currentText().strip()
        if name == "Elegir manualmente":
            QtWidgets.QMessageBox.information(self, "Restriciones de Dosis", "No hay un preset seleccionado para borrar.")
            return
        if _is_builtin_constraint_preset(name):
            QtWidgets.QMessageBox.warning(self, "Restriciones de Dosis", "No se puede borrar un preset incorporado.")
            return
        if QtWidgets.QMessageBox.question(self, "Borrar preset", f"¿Borrar el preset '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        CONSTRAINT_PRESETS.pop(name, None)
        self._selected_name = ""
        self._reload_combo()
        self.cmb_preset.setCurrentText("Elegir manualmente")
        self.lbl_info.setText("Preset borrado.")

    def _accept(self):
        try:
            self._read_matrix()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Restriciones de Dosis", str(e))
            return
        self.accept()

    def export_to(self, tbl_cons: QtWidgets.QTableWidget):
        for i in range(self.tbl.rowCount()):
            for j in range(self.tbl.columnCount()):
                it = self.tbl.item(i, j)
                txt = it.text().strip() if it else "0"
                tbl_cons.setItem(i, j, QtWidgets.QTableWidgetItem(txt if txt else "0"))

    def selected_preset_name(self) -> str:
        return self._selected_name.strip()

