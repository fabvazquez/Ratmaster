"""
ui/dialogs/boro.py
==================
Diálogo de configuración del protocolo de boro (BPA, GB10, o manual).

Permite seleccionar un preset de la librería interna o definir concentraciones
de boro por órgano manualmente con sus incertidumbres.
"""

import json
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.constants import ORG_ORDER, PROTO_LIB, USER_BORO_PROTOCOLS
from ratmaster.constants import (
    _is_builtin_boro_protocol, _resolve_boro_protocol_name,
    _sanitize_boro_protocol_dict,
)
from ratmaster.app_paths import ensure_user_json


class BoroDialog(QtWidgets.QDialog):
    def __init__(self, parent, tbl_boro, current_name=""):
        super().__init__(parent)
        self.setWindowTitle("Protocolo de Boro")
        self.resize(980, 280)
        self._selected_name = current_name.strip() if current_name else "(manual)"
        self._loading = False

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Protocolo:"))
        self.cmb_proto = QtWidgets.QComboBox()
        self._reload_combo()
        top.addWidget(self.cmb_proto, 1)

        self.btn_save = QtWidgets.QPushButton("Guardar como preset…")
        self.btn_delete = QtWidgets.QPushButton("Borrar preset")
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_delete)
        layout.addLayout(top)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color:#546E7A; font-size:9pt;")
        layout.addWidget(self.lbl_info)

        self.tbl = QtWidgets.QTableWidget(2, len(ORG_ORDER))
        self.tbl.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl.setVerticalHeaderLabels(["[B] (ppm)", "Error [B] (ppm)"])
        layout.addWidget(self.tbl)

        btns = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("Aceptar")
        btn_cancel = QtWidgets.QPushButton("Cancelar")
        btns.addStretch()
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        self.btn_save.clicked.connect(self._save_as_preset)
        self.btn_delete.clicked.connect(self._delete_current_preset)
        self.cmb_proto.currentTextChanged.connect(self._on_combo_changed)
        self.tbl.itemChanged.connect(self._on_table_edited)
        btn_ok.clicked.connect(self._accept)
        btn_cancel.clicked.connect(self.reject)

        # inicializar tabla
        self._copy_from(tbl_boro)
        if current_name and current_name in [self.cmb_proto.itemText(i) for i in range(self.cmb_proto.count())]:
            self.cmb_proto.setCurrentText(current_name)
        else:
            self.cmb_proto.setCurrentText("Editar manualmente")
            self._selected_name = "(manual)"
            self.lbl_info.setText("Protocolo manual.")

    def _reload_combo(self):
        cur = self._selected_name
        self.cmb_proto.clear()
        self.cmb_proto.addItem("Editar manualmente")
        for name in sorted(PROTO_LIB.keys()):
            self.cmb_proto.addItem(name)
        if cur and cur != "(manual)":
            idx = self.cmb_proto.findText(cur)
            if idx >= 0:
                self.cmb_proto.setCurrentIndex(idx)

    def _copy_from(self, tbl_boro):
        self._loading = True
        try:
            for i in range(2):
                for j in range(len(ORG_ORDER)):
                    it = tbl_boro.item(i, j)
                    txt = it.text().strip() if it else "0"
                    self.tbl.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        finally:
            self._loading = False

    def _set_table_from_protocol(self, name):
        lib = PROTO_LIB.get(name)
        if not lib:
            return
        self._loading = True
        try:
            for j in range(len(ORG_ORDER)):
                self.tbl.setItem(0, j, QtWidgets.QTableWidgetItem(f"{float(lib['B_ppm'][j]):.6g}"))
                self.tbl.setItem(1, j, QtWidgets.QTableWidgetItem(f"{float(lib['B_err'][j]):.6g}"))
        finally:
            self._loading = False
        self._selected_name = name
        self.lbl_info.setText(f"Protocolo seleccionado: {name}")

    def _on_combo_changed(self, text):
        text = (text or "").strip()
        if text == "Editar manualmente":
            self._selected_name = "(manual)"
            self.lbl_info.setText("Protocolo manual.")
            return
        self._set_table_from_protocol(text)

    def _on_table_edited(self, item=None):
        if self._loading:
            return
        blocker = QtCore.QSignalBlocker(self.cmb_proto)
        self.cmb_proto.setCurrentText("Editar manualmente")
        del blocker
        self._selected_name = "(manual)"
        self.lbl_info.setText("Protocolo manual.")

    def _table_arrays(self):
        B, Be = [], []
        for j in range(len(ORG_ORDER)):
            def f(r):
                it = self.tbl.item(r, j)
                txt = (it.text() if it else "0").strip().replace(",", ".")
                return float(txt) if txt else 0.0
            b = f(0); be = f(1)
            if b < 0 or be < 0:
                raise ValueError(f"No se permiten valores negativos en {ORG_ORDER[j]}.")
            B.append(b); Be.append(be)
        return np.array(B, dtype=float), np.array(Be, dtype=float)

    def _save_as_preset(self):
        try:
            B, Be = self._table_arrays()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Protocolo", str(e))
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Guardar protocolo", "Nombre del nuevo protocolo:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in PROTO_LIB:
            if QtWidgets.QMessageBox.question(self, "Sobrescribir", f"Ya existe '{name}'. ¿Sobrescribirlo?") != QtWidgets.QMessageBox.Yes:
                return
        PROTO_LIB[name] = {"ref": "Usuario", "B_ppm": B.tolist(), "B_err": Be.tolist()}
        self._selected_name = name
        self._reload_combo()
        self.cmb_proto.setCurrentText(name)
        self.lbl_info.setText(f"Protocolo guardado: {name}")

    def _delete_current_preset(self):
        name = self.cmb_proto.currentText().strip()
        if name == "Editar manualmente":
            QtWidgets.QMessageBox.information(self, "Protocolo", "No hay un preset seleccionado para borrar.")
            return
        if _is_builtin_boro_protocol(name):
            QtWidgets.QMessageBox.warning(self, "Protocolo", "No se puede borrar un protocolo incorporado.")
            return
        if QtWidgets.QMessageBox.question(self, "Borrar protocolo", f"¿Borrar el protocolo '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        PROTO_LIB.pop(name, None)
        self._selected_name = "(manual)"
        self._reload_combo()
        self.cmb_proto.setCurrentText("Editar manualmente")
        self.lbl_info.setText("Protocolo borrado.")

    def _accept(self):
        try:
            self._table_arrays()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Protocolo", str(e))
            return
        self.accept()

    def selected_protocol_name(self):
        return self._selected_name

