"""
ui/dialogs/bio_params.py
========================
Diálogo para visualizar y editar los parámetros biológicos CBE y RBE
antes de ejecutar el cálculo de dosis equivalente.

CBE (Compound Biological Effectiveness): factor multiplicativo para el boro.
RBE (Relative Biological Effectiveness): factor multiplicativo para neutrones.

Igual que con los protocolos de boro (ver ui/dialogs/boro.py), este diálogo
permite elegir un preset de la librería BIO_LIB (p. ej. "Convencional
(glioblastoma)" o "Calculado (tejidos específicos)"), editar los valores a
mano, y guardar la combinación actual como un preset nuevo con
"Guardar como preset…" — queda disponible para elegir la próxima vez y
persiste en disco (%APPDATA%\\RatMaster\\user_bio_presets.json).
"""

import numpy as np
from PySide6 import QtCore, QtWidgets

from ratmaster.constants import ORG_ORDER, BIO_LIB, USER_BIO_PRESETS
from ratmaster.constants import _is_builtin_bio_preset, organ_array_to_dict
from ratmaster.data.persistence import save_user_bio_presets


class BioParamsDialog(QtWidgets.QDialog):
    """
    Muestra las tablas de CBE y RBE del cálculo actual, permitiendo elegir un
    preset guardado, editar los valores a mano, y guardar la combinación
    actual como preset nuevo.

    La tabla se pre-pobla con los valores actuales de la ventana principal
    (tbl_cbe, tbl_rbe) y los cambios se aplican solo al aceptar.
    """

    def __init__(
        self,
        parent,
        tbl_cbe: QtWidgets.QTableWidget,
        tbl_rbe: QtWidgets.QTableWidget,
        current_name: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Parámetros biológicos (CBE / RBE)")
        self.resize(900, 380)
        self._selected_name = current_name.strip() if current_name else "(manual)"
        self._loading = False

        layout = QtWidgets.QVBoxLayout(self)

        # ── Selector de preset ───────────────────────────────────────────────
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Preset:"))
        self.cmb_preset = QtWidgets.QComboBox()
        self._reload_combo()
        top.addWidget(self.cmb_preset, 1)

        self.btn_save = QtWidgets.QPushButton("Guardar como preset…")
        self.btn_delete = QtWidgets.QPushButton("Borrar preset")
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_delete)
        layout.addLayout(top)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color:#546E7A; font-size:9pt;")
        layout.addWidget(self.lbl_info)

        # ── Tabla CBE ────────────────────────────────────────────────────────
        self.tbl_cbe = QtWidgets.QTableWidget(1, len(ORG_ORDER))
        self.tbl_cbe.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_cbe.setVerticalHeaderLabels(["CBE"])

        # ── Tabla RBE ────────────────────────────────────────────────────────
        self.tbl_rbe = QtWidgets.QTableWidget(1, len(ORG_ORDER))
        self.tbl_rbe.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_rbe.setVerticalHeaderLabels(["RBE"])

        layout.addWidget(QtWidgets.QLabel("CBE (Compound Biological Effectiveness)"))
        layout.addWidget(self.tbl_cbe)
        layout.addWidget(QtWidgets.QLabel("RBE (Relative Biological Effectiveness)"))
        layout.addWidget(self.tbl_rbe)

        # ── Botones ──────────────────────────────────────────────────────────
        btns       = QtWidgets.QHBoxLayout()
        btn_ok     = QtWidgets.QPushButton("Aceptar")
        btn_cancel = QtWidgets.QPushButton("Cancelar")
        btns.addStretch()
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        self.btn_save.clicked.connect(self._save_as_preset)
        self.btn_delete.clicked.connect(self._delete_current_preset)
        self.cmb_preset.currentTextChanged.connect(self._on_combo_changed)
        self.tbl_cbe.itemChanged.connect(self._on_table_edited)
        self.tbl_rbe.itemChanged.connect(self._on_table_edited)
        btn_ok.clicked.connect(self._accept)
        btn_cancel.clicked.connect(self.reject)

        # inicializar tablas con los valores actuales de la ventana principal
        self._copy_from(tbl_cbe, tbl_rbe)
        if current_name and current_name in [self.cmb_preset.itemText(i) for i in range(self.cmb_preset.count())]:
            self.cmb_preset.setCurrentText(current_name)
            self._set_tables_from_preset(current_name)
        else:
            self.cmb_preset.setCurrentText("Editar manualmente")
            self._selected_name = "(manual)"
            self.lbl_info.setText("Preset manual.")

    # ── Combo de presets ──────────────────────────────────────────────────────

    def _reload_combo(self):
        cur = self._selected_name
        self.cmb_preset.clear()
        self.cmb_preset.addItem("Editar manualmente")
        for name in sorted(BIO_LIB.keys()):
            self.cmb_preset.addItem(name)
        if cur and cur != "(manual)":
            idx = self.cmb_preset.findText(cur)
            if idx >= 0:
                self.cmb_preset.setCurrentIndex(idx)

    def _copy_from(self, tbl_cbe, tbl_rbe):
        self._loading = True
        try:
            for j in range(tbl_cbe.columnCount()):
                item = tbl_cbe.item(0, j)
                self.tbl_cbe.setItem(0, j, QtWidgets.QTableWidgetItem(item.text() if item else "1.0"))
            for j in range(tbl_rbe.columnCount()):
                item = tbl_rbe.item(0, j)
                self.tbl_rbe.setItem(0, j, QtWidgets.QTableWidgetItem(item.text() if item else "1.0"))
        finally:
            self._loading = False

    def _set_tables_from_preset(self, name):
        lib = BIO_LIB.get(name)
        if not lib:
            return
        rbe_raw = lib.get("RBE_value", {}) or {}
        cbe_raw = lib.get("CBE_value", {}) or {}
        self._loading = True
        try:
            for j, organ in enumerate(ORG_ORDER):
                r = float(rbe_raw.get(organ, 1.0))
                c = float(cbe_raw.get(organ, 1.0))
                self.tbl_rbe.setItem(0, j, QtWidgets.QTableWidgetItem(f"{r:.6g}"))
                self.tbl_cbe.setItem(0, j, QtWidgets.QTableWidgetItem(f"{c:.6g}"))
        finally:
            self._loading = False
        self._selected_name = name
        self.lbl_info.setText(f"Preset seleccionado: {name}")

    def _on_combo_changed(self, text):
        text = (text or "").strip()
        if text == "Editar manualmente":
            self._selected_name = "(manual)"
            self.lbl_info.setText("Preset manual.")
            return
        self._set_tables_from_preset(text)

    def _on_table_edited(self, item=None):
        if self._loading:
            return
        blocker = QtCore.QSignalBlocker(self.cmb_preset)
        self.cmb_preset.setCurrentText("Editar manualmente")
        del blocker
        self._selected_name = "(manual)"
        self.lbl_info.setText("Preset manual.")

    # ── Helpers numéricos ────────────────────────────────────────────────────

    def _table_arrays(self):
        CBE, RBE = [], []
        for j in range(len(ORG_ORDER)):
            def f(table, default=1.0):
                it = table.item(0, j)
                txt = (it.text() if it else str(default)).strip().replace(",", ".")
                return float(txt) if txt else default
            c = f(self.tbl_cbe)
            r = f(self.tbl_rbe)
            if c < 0 or r < 0:
                raise ValueError(f"No se permiten valores negativos en {ORG_ORDER[j]}.")
            CBE.append(c); RBE.append(r)
        return np.array(CBE, dtype=float), np.array(RBE, dtype=float)

    # ── Guardar / borrar preset ──────────────────────────────────────────────

    def _save_as_preset(self):
        try:
            CBE, RBE = self._table_arrays()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Preset biológico", str(e))
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Guardar preset", "Nombre del nuevo preset:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in BIO_LIB:
            if QtWidgets.QMessageBox.question(self, "Sobrescribir", f"Ya existe '{name}'. ¿Sobrescribirlo?") != QtWidgets.QMessageBox.Yes:
                return
        entry = {
            "ref": "Usuario",
            "RBE_value": organ_array_to_dict(RBE, ORG_ORDER),
            "CBE_value": organ_array_to_dict(CBE, ORG_ORDER),
        }
        BIO_LIB[name] = entry
        USER_BIO_PRESETS[name] = entry
        ok_disk, err = save_user_bio_presets(USER_BIO_PRESETS)
        if not ok_disk:
            QtWidgets.QMessageBox.warning(self, "Preset biológico", f"No se pudo guardar en disco:\n{err}")
        self._selected_name = name
        self._reload_combo()
        self.cmb_preset.setCurrentText(name)
        self.lbl_info.setText(f"Preset guardado: {name}")

    def _delete_current_preset(self):
        name = self.cmb_preset.currentText().strip()
        if name == "Editar manualmente":
            QtWidgets.QMessageBox.information(self, "Preset biológico", "No hay un preset seleccionado para borrar.")
            return
        if _is_builtin_bio_preset(name):
            QtWidgets.QMessageBox.warning(self, "Preset biológico", "No se puede borrar un preset incorporado.")
            return
        if QtWidgets.QMessageBox.question(self, "Borrar preset", f"¿Borrar el preset '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        BIO_LIB.pop(name, None)
        USER_BIO_PRESETS.pop(name, None)
        ok_disk, err = save_user_bio_presets(USER_BIO_PRESETS)
        if not ok_disk:
            QtWidgets.QMessageBox.warning(self, "Preset biológico", f"No se pudo guardar en disco:\n{err}")
        self._selected_name = "(manual)"
        self._reload_combo()
        self.cmb_preset.setCurrentText("Editar manualmente")
        self.lbl_info.setText("Preset borrado.")

    # ── Aceptar / resultado ──────────────────────────────────────────────────

    def _accept(self):
        try:
            self._table_arrays()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Preset biológico", str(e))
            return
        self.accept()

    def selected_preset_name(self):
        return self._selected_name
