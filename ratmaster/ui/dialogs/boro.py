"""
ui/dialogs/boro.py
==================
Diálogo de configuración del protocolo de boro (BPA, GB10, o manual).

Permite seleccionar un preset de la librería interna o definir concentraciones
de boro por órgano manualmente con sus incertidumbres.

Desde esta versión, además del modo "genérico" (los valores de B_ppm/B_err
del protocolo tal cual están en la librería), soporta un modo "a partir de
sangre": el usuario ingresa la concentración de boro medida en sangre del
animal, y la concentración en cada órgano se calcula multiplicando esa
medición por la relación Tejido/Sangre (TB_ratio) definida para ese
protocolo. Esta relación es distinta para cada protocolo (BPA, GB-10, etc.),
así que se guarda como parte del preset (TB_ratio/TB_ratio_err), igual que
B_ppm/B_err.

Propagación de incertidumbre (ratio y medición de sangre se asumen
independientes):
    B_organo     = TB_ratio × [B]_sangre
    err_organo   = sqrt( (err_TB_ratio × [B]_sangre)² + (TB_ratio × err_sangre)² )
"""

import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.constants import ORG_ORDER, PROTO_LIB, USER_BORO_PROTOCOLS
from ratmaster.constants import (
    _is_builtin_boro_protocol, _resolve_boro_protocol_name,
    _sanitize_boro_protocol_dict, organ_array_to_dict, protocol_has_blood_ratio,
)
from ratmaster.data.persistence import save_user_boro_protocols


class BoroDialog(QtWidgets.QDialog):
    def __init__(self, parent, tbl_boro, current_name="",
                 source_mode="generico", blood_conc=None, blood_conc_err=None):
        super().__init__(parent)
        self.setWindowTitle("Protocolo de Boro")
        self.resize(980, 420)
        self._selected_name = current_name.strip() if current_name else "(manual)"
        self._loading = False
        self._mode = "sangre" if (source_mode or "").strip().lower() == "sangre" else "generico"

        # Snapshots de los valores "genéricos" (B_ppm/B_err) y de la relación
        # Tejido/Sangre (TB_ratio/TB_ratio_err) actualmente cargados/editados.
        # Se mantienen aparte de self.tbl porque, en modo sangre, self.tbl
        # muestra valores CALCULADOS (no editables a mano), y no queremos
        # perder los valores genéricos al guardar el preset.
        self._generic_B: dict = {o: 0.0 for o in ORG_ORDER}
        self._generic_Be: dict = {o: 0.0 for o in ORG_ORDER}
        self._ratio: dict = {o: 0.0 for o in ORG_ORDER}
        self._ratio_err: dict = {o: 0.0 for o in ORG_ORDER}

        layout = QtWidgets.QVBoxLayout(self)

        # ── Selector de protocolo ────────────────────────────────────────────
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

        # ── Origen de la concentración de boro ──────────────────────────────
        box_origin = QtWidgets.QGroupBox("Origen de la concentración de boro")
        lay_origin = QtWidgets.QVBoxLayout(box_origin)

        rb_row = QtWidgets.QHBoxLayout()
        self.rb_generico = QtWidgets.QRadioButton("Genérico (valores del protocolo)")
        self.rb_sangre = QtWidgets.QRadioButton("A partir de concentración en sangre del animal")
        rb_row.addWidget(self.rb_generico)
        rb_row.addWidget(self.rb_sangre)
        rb_row.addStretch()
        lay_origin.addLayout(rb_row)

        self.lbl_blood_warn = QtWidgets.QLabel("")
        self.lbl_blood_warn.setStyleSheet("color:#B00020; font-size:9pt;")
        self.lbl_blood_warn.setWordWrap(True)
        lay_origin.addWidget(self.lbl_blood_warn)

        blood_row = QtWidgets.QHBoxLayout()
        blood_row.addWidget(QtWidgets.QLabel("[B] en sangre (ppm):"))
        self.in_blood = QtWidgets.QLineEdit()
        self.in_blood.setPlaceholderText("ej. 25.4")
        blood_row.addWidget(self.in_blood)
        blood_row.addWidget(QtWidgets.QLabel("Error (ppm):"))
        self.in_blood_err = QtWidgets.QLineEdit()
        self.in_blood_err.setPlaceholderText("ej. 1.2")
        blood_row.addWidget(self.in_blood_err)
        blood_row.addStretch()
        self.widget_blood_row = QtWidgets.QWidget()
        self.widget_blood_row.setLayout(blood_row)
        lay_origin.addWidget(self.widget_blood_row)

        layout.addWidget(box_origin)

        # ── Tabla de relación Tejido/Sangre (solo en modo "sangre") ─────────
        calc_row = QtWidgets.QHBoxLayout()
        calc_row.addWidget(QtWidgets.QLabel("Sangre de referencia del protocolo (ppm):"))
        self.in_ref_blood = QtWidgets.QLineEdit()
        self.in_ref_blood.setPlaceholderText("ej. 13.7")
        calc_row.addWidget(self.in_ref_blood)
        calc_row.addWidget(QtWidgets.QLabel("Error:"))
        self.in_ref_blood_err = QtWidgets.QLineEdit()
        self.in_ref_blood_err.setPlaceholderText("ej. 2.3")
        calc_row.addWidget(self.in_ref_blood_err)
        self.btn_calc_ratio = QtWidgets.QPushButton("Calcular relación T/S desde la tabla de concentración ↓")
        calc_row.addWidget(self.btn_calc_ratio)
        calc_row.addStretch()
        self.widget_calc_ratio_row = QtWidgets.QWidget()
        self.widget_calc_ratio_row.setLayout(calc_row)
        lbl_calc_help = QtWidgets.QLabel(
            "Si ya conocés la concentración tisular de cada órgano y la concentración en sangre "
            "de esa misma medición (p. ej. de un paper), cargá esos valores como \"[B] (ppm)\" más "
            "abajo, poné acá la sangre de esa medición, y usá este botón para que la relación T/S "
            "se calcule sola en vez de tener que dividir a mano."
        )
        lbl_calc_help.setStyleSheet("color:#546E7A; font-size:8pt;")
        lbl_calc_help.setWordWrap(True)
        self.widget_calc_ratio_row.layout().setContentsMargins(0, 0, 0, 0)
        calc_box = QtWidgets.QVBoxLayout()
        calc_box.setContentsMargins(0, 0, 0, 0)
        calc_box.addWidget(lbl_calc_help)
        calc_box.addWidget(self.widget_calc_ratio_row)
        self.widget_calc_ratio = QtWidgets.QWidget()
        self.widget_calc_ratio.setLayout(calc_box)
        layout.addWidget(self.widget_calc_ratio)

        self.tbl_ratio = QtWidgets.QTableWidget(2, len(ORG_ORDER))
        self.tbl_ratio.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_ratio.setVerticalHeaderLabels(["Relación T/S", "Error relación T/S"])
        layout.addWidget(self.tbl_ratio)

        # ── Tabla de concentración de boro por órgano (ppm) ─────────────────
        # En modo genérico: editable a mano.
        # En modo sangre: de solo lectura, calculada a partir de la tabla de
        # relación T/S y la concentración en sangre.
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
        self.tbl_ratio.itemChanged.connect(self._on_ratio_edited)
        self.rb_generico.toggled.connect(self._on_mode_toggled)
        self.in_blood.textChanged.connect(self._on_blood_edited)
        self.in_blood_err.textChanged.connect(self._on_blood_edited)
        self.btn_calc_ratio.clicked.connect(self._calc_ratio_from_generic)
        btn_ok.clicked.connect(self._accept)
        btn_cancel.clicked.connect(self.reject)

        # inicializar tabla genérica y de relación T/S
        self._copy_from(tbl_boro)
        if current_name and current_name in [self.cmb_proto.itemText(i) for i in range(self.cmb_proto.count())]:
            self.cmb_proto.setCurrentText(current_name)
            self._set_table_from_protocol(current_name)
        else:
            self.cmb_proto.setCurrentText("Editar manualmente")
            self._selected_name = "(manual)"
            self.lbl_info.setText("Protocolo manual.")

        # inicializar modo e inputs de sangre
        if blood_conc is not None:
            self.in_blood.setText(f"{float(blood_conc):.6g}")
        if blood_conc_err is not None:
            self.in_blood_err.setText(f"{float(blood_conc_err):.6g}")
        if self._mode == "sangre":
            self.rb_sangre.setChecked(True)
        else:
            self.rb_generico.setChecked(True)
        self._on_mode_toggled()

    # ── Combo de protocolos ──────────────────────────────────────────────────

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
                    organ = ORG_ORDER[j]
                    val = self._safe_float(txt)
                    if i == 0:
                        self._generic_B[organ] = val
                    else:
                        self._generic_Be[organ] = val
        finally:
            self._loading = False

    def _set_table_from_protocol(self, name):
        lib = PROTO_LIB.get(name)
        if not lib:
            return
        b_raw   = lib.get("B_ppm", {}) or {}
        be_raw  = lib.get("B_err", {}) or {}
        tb_raw  = lib.get("TB_ratio", {}) or {}
        tbe_raw = lib.get("TB_ratio_err", {}) or {}
        # Tolerancia: si por algún motivo quedó en memoria el formato
        # posicional viejo (lista en vez de dict), se interpreta alineado
        # a ORG_ORDER actual en vez de romper. El formato esperado de aquí
        # en adelante es siempre dict {organo: valor}.
        if not isinstance(b_raw, dict):
            b_raw = {o: v for o, v in zip(ORG_ORDER, b_raw)}
        if not isinstance(be_raw, dict):
            be_raw = {o: v for o, v in zip(ORG_ORDER, be_raw)}

        self._loading = True
        try:
            for j, organ in enumerate(ORG_ORDER):
                b   = float(b_raw.get(organ, 0.0))
                be  = float(be_raw.get(organ, 0.0))
                tb  = float(tb_raw.get(organ, 0.0))
                tbe = float(tbe_raw.get(organ, 0.0))
                self._generic_B[organ] = b
                self._generic_Be[organ] = be
                self._ratio[organ] = tb
                self._ratio_err[organ] = tbe
                self.tbl.setItem(0, j, QtWidgets.QTableWidgetItem(f"{b:.6g}"))
                self.tbl.setItem(1, j, QtWidgets.QTableWidgetItem(f"{be:.6g}"))
                self.tbl_ratio.setItem(0, j, QtWidgets.QTableWidgetItem(f"{tb:.6g}"))
                self.tbl_ratio.setItem(1, j, QtWidgets.QTableWidgetItem(f"{tbe:.6g}"))
        finally:
            self._loading = False
        self._selected_name = name
        self.lbl_info.setText(f"Protocolo seleccionado: {name}")
        ref_blood = float(lib.get("ref_blood_conc", 0.0) or 0.0)
        ref_blood_err = float(lib.get("ref_blood_conc_err", 0.0) or 0.0)
        self.in_ref_blood.setText(f"{ref_blood:.6g}" if ref_blood else "")
        self.in_ref_blood_err.setText(f"{ref_blood_err:.6g}" if ref_blood_err else "")
        self._update_blood_warning()
        if self._mode == "sangre":
            self._recompute_from_blood()

    def _on_combo_changed(self, text):
        text = (text or "").strip()
        if text == "Editar manualmente":
            self._selected_name = "(manual)"
            self.lbl_info.setText("Protocolo manual.")
            self._update_blood_warning()
            return
        self._set_table_from_protocol(text)

    # ── Modo genérico / sangre ───────────────────────────────────────────────

    def _on_mode_toggled(self, *_args):
        self._mode = "sangre" if self.rb_sangre.isChecked() else "generico"
        is_sangre = self._mode == "sangre"

        self.widget_blood_row.setVisible(is_sangre)
        self.tbl_ratio.setVisible(is_sangre)
        self.widget_calc_ratio.setVisible(is_sangre)
        self.tbl.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers if is_sangre
            else QtWidgets.QAbstractItemView.AllEditTriggers
        )

        if is_sangre:
            self._update_blood_warning()
            self._recompute_from_blood()
        else:
            self.lbl_blood_warn.setText("")
            self._write_generic_to_table()

    def _update_blood_warning(self):
        if self._mode != "sangre":
            self.lbl_blood_warn.setText("")
            return
        has_ratio = any(float(v) != 0.0 for v in self._ratio.values())
        if not has_ratio:
            name = self._selected_name
            extra = f"El protocolo '{name}' todavía no tiene" if name and name != "(manual)" else "Todavía no hay"
            self.lbl_blood_warn.setText(
                f"{extra} una relación Tejido/Sangre cargada. Completá la tabla de abajo a mano, "
                f"o cargá la concentración por órgano + la sangre de referencia de esa medición y "
                f"usá \"Calcular relación T/S…\" para que se calcule sola."
            )
        else:
            self.lbl_blood_warn.setText("")

    def _write_generic_to_table(self):
        self._loading = True
        try:
            for j, organ in enumerate(ORG_ORDER):
                b = self._generic_B.get(organ, 0.0)
                be = self._generic_Be.get(organ, 0.0)
                self.tbl.setItem(0, j, QtWidgets.QTableWidgetItem(f"{b:.6g}"))
                self.tbl.setItem(1, j, QtWidgets.QTableWidgetItem(f"{be:.6g}"))
        finally:
            self._loading = False

    def _blood_values(self):
        conc = self._safe_float(self.in_blood.text())
        err = self._safe_float(self.in_blood_err.text())
        return conc, err

    def _ref_blood_values(self):
        conc = self._safe_float(self.in_ref_blood.text())
        err = self._safe_float(self.in_ref_blood_err.text())
        return conc, err

    def _calc_ratio_from_generic(self):
        """
        Calcula la relación Tejido/Sangre dividiendo la tabla de concentración
        genérica actual (self._generic_B/_generic_Be, la de arriba/abajo según
        el layout — la tabla "[B] (ppm)") por la concentración en sangre de
        referencia ingresada (la de esa misma medición/paper), en vez de
        obligar a tipear la relación a mano.

            TB_ratio     = B_organo / B_sangre_ref
            err_TB_ratio = TB_ratio × sqrt((err_organo/B_organo)² + (err_sangre_ref/B_sangre_ref)²)
        """
        cb, cb_err = self._ref_blood_values()
        if cb <= 0:
            QtWidgets.QMessageBox.warning(
                self, "Relación Tejido/Sangre",
                "Ingresá una concentración en sangre de referencia mayor a cero "
                "(la de la misma medición/publicación que la tabla de concentración por órgano)."
            )
            return
        self._loading = True
        try:
            for j, organ in enumerate(ORG_ORDER):
                b = self._generic_B.get(organ, 0.0)
                be = self._generic_Be.get(organ, 0.0)
                if b <= 0:
                    r, re = 0.0, 0.0
                else:
                    r = b / cb
                    re = r * ((be / b) ** 2 + (cb_err / cb) ** 2) ** 0.5
                self._ratio[organ] = r
                self._ratio_err[organ] = re
                self.tbl_ratio.setItem(0, j, QtWidgets.QTableWidgetItem(f"{r:.6g}"))
                self.tbl_ratio.setItem(1, j, QtWidgets.QTableWidgetItem(f"{re:.6g}"))
        finally:
            self._loading = False
        self.lbl_info.setText(
            "Relación Tejido/Sangre calculada a partir de la tabla de concentración "
            f"y una sangre de referencia de {cb:.6g} ± {cb_err:.6g} ppm."
        )
        self._update_blood_warning()
        if self._mode == "sangre":
            self._recompute_from_blood()

    def _recompute_from_blood(self):
        if self._mode != "sangre":
            return
        conc, cerr = self._blood_values()
        self._loading = True
        try:
            for j, organ in enumerate(ORG_ORDER):
                r = self._ratio.get(organ, 0.0)
                re = self._ratio_err.get(organ, 0.0)
                b = r * conc
                be = ((re * conc) ** 2 + (r * cerr) ** 2) ** 0.5
                self.tbl.setItem(0, j, QtWidgets.QTableWidgetItem(f"{b:.6g}"))
                self.tbl.setItem(1, j, QtWidgets.QTableWidgetItem(f"{be:.6g}"))
        finally:
            self._loading = False

    def _on_blood_edited(self, *_args):
        if self._loading or self._mode != "sangre":
            return
        self._recompute_from_blood()

    def _on_ratio_edited(self, item=None):
        if self._loading or self._mode != "sangre":
            return
        for j, organ in enumerate(ORG_ORDER):
            self._ratio[organ] = self._cell_float(self.tbl_ratio, 0, j)
            self._ratio_err[organ] = self._cell_float(self.tbl_ratio, 1, j)
        # Editar la relación a mano desliga el preset seleccionado, igual que
        # editar la tabla de concentración en modo genérico.
        blocker = QtCore.QSignalBlocker(self.cmb_proto)
        self.cmb_proto.setCurrentText("Editar manualmente")
        del blocker
        self._selected_name = "(manual)"
        self.lbl_info.setText("Protocolo manual.")
        self._recompute_from_blood()

    def _on_table_edited(self, item=None):
        if self._loading:
            return
        if self._mode == "generico":
            for j, organ in enumerate(ORG_ORDER):
                self._generic_B[organ] = self._cell_float(self.tbl, 0, j)
                self._generic_Be[organ] = self._cell_float(self.tbl, 1, j)
        blocker = QtCore.QSignalBlocker(self.cmb_proto)
        self.cmb_proto.setCurrentText("Editar manualmente")
        del blocker
        self._selected_name = "(manual)"
        self.lbl_info.setText("Protocolo manual.")

    # ── Helpers numéricos ────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(txt, default=0.0):
        txt = (txt or "").strip().replace(",", ".")
        try:
            return float(txt) if txt else default
        except Exception:
            return default

    def _cell_float(self, table, row, col, default=0.0):
        it = table.item(row, col)
        return self._safe_float(it.text() if it else "", default)

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

    def _ratio_arrays(self):
        R, Re = [], []
        for j in range(len(ORG_ORDER)):
            r = self._cell_float(self.tbl_ratio, 0, j)
            re = self._cell_float(self.tbl_ratio, 1, j)
            if r < 0 or re < 0:
                raise ValueError(f"No se permiten valores negativos en la relación T/S de {ORG_ORDER[j]}.")
            R.append(r); Re.append(re)
        return np.array(R, dtype=float), np.array(Re, dtype=float)

    # ── Guardar / borrar preset ──────────────────────────────────────────────

    def _save_as_preset(self):
        try:
            # Los valores "genéricos" del preset son siempre self._generic_B/
            # _generic_Be (no self.tbl, que en modo sangre muestra valores
            # calculados, no los del protocolo).
            B = np.array([self._generic_B.get(o, 0.0) for o in ORG_ORDER], dtype=float)
            Be = np.array([self._generic_Be.get(o, 0.0) for o in ORG_ORDER], dtype=float)
            if np.any(B < 0) or np.any(Be < 0):
                raise ValueError("No se permiten valores negativos en la concentración genérica.")
            R, Re = self._ratio_arrays()
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
        # B_ppm/B_err/TB_ratio/TB_ratio_err se guardan como dict {organo: valor}
        # (no lista posicional) para que el protocolo siga siendo válido si más
        # adelante se agrega o quita un órgano de ORG_ORDER.
        ref_blood, ref_blood_err = self._ref_blood_values()
        entry = {
            "ref": "Usuario",
            "B_ppm": organ_array_to_dict(B, ORG_ORDER),
            "B_err": organ_array_to_dict(Be, ORG_ORDER),
            "TB_ratio": organ_array_to_dict(R, ORG_ORDER),
            "TB_ratio_err": organ_array_to_dict(Re, ORG_ORDER),
            # Solo informativo/trazabilidad (de dónde salió la relación T/S).
            "ref_blood_conc": ref_blood, "ref_blood_conc_err": ref_blood_err,
        }
        PROTO_LIB[name] = entry
        USER_BORO_PROTOCOLS[name] = entry
        ok_disk, err = save_user_boro_protocols(USER_BORO_PROTOCOLS)
        if not ok_disk:
            QtWidgets.QMessageBox.warning(self, "Protocolo", f"No se pudo guardar en disco:\n{err}")
        self._selected_name = name
        self._reload_combo()
        self.cmb_proto.setCurrentText(name)
        self.lbl_info.setText(f"Protocolo guardado: {name}")
        self._update_blood_warning()

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
        USER_BORO_PROTOCOLS.pop(name, None)
        ok_disk, err = save_user_boro_protocols(USER_BORO_PROTOCOLS)
        if not ok_disk:
            QtWidgets.QMessageBox.warning(self, "Protocolo", f"No se pudo guardar en disco:\n{err}")
        self._selected_name = "(manual)"
        self._reload_combo()
        self.cmb_proto.setCurrentText("Editar manualmente")
        self.lbl_info.setText("Protocolo borrado.")

    # ── Aceptar / resultado ──────────────────────────────────────────────────

    def _accept(self):
        try:
            self._table_arrays()
            if self._mode == "sangre":
                conc, cerr = self._blood_values()
                if conc < 0 or cerr < 0:
                    raise ValueError("La concentración de boro en sangre no puede ser negativa.")
                self._ratio_arrays()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Protocolo", str(e))
            return
        self.accept()

    def selected_protocol_name(self):
        return self._selected_name

    def selected_source_mode(self):
        """'generico' o 'sangre', según el origen elegido para la concentración."""
        return self._mode

    def blood_concentration(self):
        """
        (conc, err) en ppm si el modo activo es 'sangre', o (None, None) si
        el modo activo es 'generico' (no aplica).
        """
        if self._mode != "sangre":
            return None, None
        return self._blood_values()
