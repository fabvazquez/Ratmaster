"""
ui/dialogs/isoe_dialogs.py  — v4
=================================
Diálogos para el cálculo y configuración de Dosis Isoefectiva (IsoE).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLUJO DE USO DESDE LA APLICACIÓN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Al hacer clic en "Calcular IsoE":
    organs = list(report["CompDoses"].keys())
    dlg = IsoEOrganParamsDialog(parent, organs)
    if dlg.exec() != QDialog.Accepted:
        return

    params_by_organ = dlg.get_params_by_organ()
    t0_maps         = dlg.get_t0_maps_by_organ()

    # Para calcular con distintos t0_map por órgano, usar el de cada órgano
    # en un loop, o el DEFAULT_T0_MAP si el preset es biexp.
    isoe_report, metrics = compute_isoe_from_report(
        report, params_by_organ=params_by_organ
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    IsoEOrganParamsDialog — DIALOG PRINCIPAL. Asigna presets por órgano.
                            Se abre al iniciar el cálculo de IsoE.
                            Auto-detecta tejido, sugiere preset, permite editar.

    IsoEPresetsDialog     — Gestiona la biblioteca de presets.
                            Se abre desde IsoEOrganParamsDialog o el menú.

    IsoEDialog            — Editor rápido de parámetros numéricos (acceso legacy).
"""

from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.physics.isoe import (
    IsoEParams, ISOE_PARAM_PRESETS, DEFAULT_T0_MAP,
    TISSUE_CATEGORIES, REPAIR_MODEL_LABELS, PARAM_DESCRIPTIONS,
    detect_tissue_type, get_presets_for_organ, get_presets_for_tissue,
    auto_assign_presets, build_params_by_organ, get_t0_map_for_preset,
)
from ratmaster.constants import USER_ISOE_PRESETS, _is_builtin_isoe_preset
from ratmaster.data.persistence import save_user_isoe_presets


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE UI
# ══════════════════════════════════════════════════════════════════════════════

def _lbl(text: str, color: str = "#455A64",
         bold: bool = False, italic: bool = False,
         wrap: bool = False) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    s = f"color:{color};"
    if bold:   s += "font-weight:bold;"
    if italic: s += "font-style:italic;"
    w.setStyleSheet(s)
    w.setWordWrap(wrap)
    return w


def _separator() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setStyleSheet("color:#CFD8DC;")
    return f


def _all_presets() -> dict:
    return {**ISOE_PARAM_PRESETS, **USER_ISOE_PRESETS}


# ══════════════════════════════════════════════════════════════════════════════
# IsoEOrganParamsDialog — DIALOG PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class IsoEOrganParamsDialog(QtWidgets.QDialog):
    """
    Dialog principal para el cálculo de IsoE.

    Al abrir:
        - Auto-detecta el tipo de tejido de cada órgano por su nombre.
        - Asigna automáticamente el mejor preset compatible.
        - Muestra una tabla con: órgano | tejido | preset | estado.
        - El usuario puede cambiar el tejido o el preset de cada órgano.
        - Órganos sin preset válido se omitirán del cálculo.

    Al aceptar retorna params_by_organ y t0_maps listos para calcular.
    """

    def __init__(self, parent, organ_list: list[str],
                 current_assignment: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Configurar cálculo de Dosis Isoefectiva por órgano")
        self.setMinimumSize(900, 500)
        self.resize(960, 640)
        self._organs = list(organ_list)
        self._rows: list[dict] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)

        # Encabezado
        root.addWidget(_lbl(
            "<b>Dosis Isoefectiva — Asignación de parámetros por órgano</b>",
            "#1A237E", bold=True,
        ))
        root.addWidget(_lbl(
            "El sistema detectó automáticamente el tipo de tejido de cada órgano y "
            "sugirió el preset más compatible. Podés ajustar la asignación antes de calcular.",
            wrap=True,
        ))
        root.addWidget(_lbl(
            "⚠  Solo se calculará IsoE para órganos con preset asignado. "
            "No uses parámetros de tejido tumoral en órganos sanos ni viceversa.",
            "#B71C1C", bold=True, wrap=True,
        ))
        root.addWidget(_separator())

        # Botones de acción masiva
        row_actions = QtWidgets.QHBoxLayout()
        btn_auto = QtWidgets.QPushButton("🔄  Autodetectar todos")
        btn_auto.setToolTip("Asigna automáticamente el preset más compatible para cada órgano.")
        btn_auto.clicked.connect(self._autodetect_all)
        btn_clear = QtWidgets.QPushButton("✕  Limpiar todos")
        btn_clear.setToolTip("Quita el preset de todos los órganos.")
        btn_clear.clicked.connect(self._clear_all)
        btn_presets = QtWidgets.QPushButton("📚  Gestionar presets…")
        btn_presets.setToolTip("Abre el gestor de presets para agregar, editar o borrar.")
        btn_presets.clicked.connect(self._open_preset_manager)
        row_actions.addWidget(btn_auto)
        row_actions.addWidget(btn_clear)
        row_actions.addStretch()
        row_actions.addWidget(btn_presets)
        root.addLayout(row_actions)

        # Tabla de órganos con scroll
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._table_widget = QtWidgets.QWidget()
        self._grid = QtWidgets.QGridLayout(self._table_widget)
        self._grid.setSpacing(5)
        self._grid.setContentsMargins(8, 8, 8, 8)

        # Cabecera de la grilla
        headers = ["Órgano del reporte", "Tipo de tejido (detectado)",
                   "Preset / parámetros a usar", "Estado"]
        for col, h in enumerate(headers):
            lbl = _lbl(f"<b>{h}</b>", "#1A237E")
            lbl.setMinimumWidth([160, 200, 280, 200][col])
            self._grid.addWidget(lbl, 0, col)
        self._grid.addWidget(_separator(), 1, 0, 1, 4)

        for organ in self._organs:
            initial = (current_assignment or {}).get(organ)
            self._add_row(organ, initial)

        self._grid.setRowStretch(len(self._organs) + 2, 1)
        scroll.setWidget(self._table_widget)
        root.addWidget(scroll, 1)

        # Resumen
        self._lbl_summary = _lbl("", "#455A64", wrap=True)
        root.addWidget(self._lbl_summary)
        self._update_summary()

        root.addWidget(_separator())
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # Auto-detectar al abrir (si no hay asignación previa)
        if not current_assignment:
            self._autodetect_all()

    # ── Construcción de filas ──────────────────────────────────────────────────

    def _add_row(self, organ: str, initial_preset: str | None):
        row_idx = len(self._rows) + 2  # +2 por cabecera y separador

        # Col 0: nombre del órgano
        lbl_organ = _lbl(f"<b>{organ}</b>")
        self._grid.addWidget(lbl_organ, row_idx, 0)

        # Col 1: tipo de tejido
        cmb_tissue = QtWidgets.QComboBox()
        cmb_tissue.setMinimumWidth(200)
        for tt, info in TISSUE_CATEGORIES.items():
            cmb_tissue.addItem(info["label"], tt)
        detected = detect_tissue_type(organ)
        idx_t = cmb_tissue.findData(detected)
        cmb_tissue.setCurrentIndex(max(idx_t, 0))
        cmb_tissue.setToolTip(
            f"Tipo de tejido detectado automáticamente: «{detected}».\n"
            "Podés cambiarlo si la detección es incorrecta."
        )
        self._grid.addWidget(cmb_tissue, row_idx, 1)

        # Col 2: selector de preset
        cmb_preset = QtWidgets.QComboBox()
        cmb_preset.setMinimumWidth(280)
        self._rebuild_preset_combo(cmb_preset, detected, initial_preset)
        self._grid.addWidget(cmb_preset, row_idx, 2)

        # Col 3: estado
        lbl_state = _lbl("", "#455A64", wrap=True)
        lbl_state.setMinimumWidth(200)
        lbl_state.setMaximumWidth(300)
        self._grid.addWidget(lbl_state, row_idx, 3)

        entry = {
            "organ":      organ,
            "cmb_tissue": cmb_tissue,
            "cmb_preset": cmb_preset,
            "lbl_state":  lbl_state,
        }
        self._rows.append(entry)
        self._refresh_state(entry)

        cmb_tissue.currentIndexChanged.connect(
            lambda _, e=entry: self._on_tissue_changed(e)
        )
        cmb_preset.currentIndexChanged.connect(
            lambda _, e=entry: (self._refresh_state(e), self._update_summary())
        )

    def _rebuild_preset_combo(self, cmb: QtWidgets.QComboBox,
                              tissue_type: str, current: str | None):
        cmb.blockSignals(True)
        cmb.clear()
        cmb.addItem("(sin asignar — no se calculará IsoE)", None)

        all_p = _all_presets()
        # Primero los de tissue_type coincidente, luego el resto
        matching    = [n for n, p in all_p.items()
                       if n != "Manual" and p.get("tissue_type") == tissue_type]
        nonmatching = [n for n, p in all_p.items()
                       if n != "Manual" and p.get("tissue_type") != tissue_type]

        if matching:
            cmb.insertSeparator(cmb.count())
            cmb.addItem(f"── Compatibles con {TISSUE_CATEGORIES.get(tissue_type,{}).get('label',tissue_type)} ──")
            cmb.model().item(cmb.count()-1).setEnabled(False)
            for name in matching:
                tt = all_p[name].get("tissue_type", "unknown")
                tt_lbl = TISSUE_CATEGORIES.get(tt, {}).get("label", tt)
                rm_key = (all_p[name].get("params") or {}).get("repair_model", "monoexp")
                rm_lbl = REPAIR_MODEL_LABELS.get(rm_key, rm_key)
                cmb.addItem(f"{name}  [{tt_lbl} · {rm_lbl}]", name)

        if nonmatching:
            cmb.insertSeparator(cmb.count())
            cmb.addItem("── Otros presets disponibles ──")
            cmb.model().item(cmb.count()-1).setEnabled(False)
            for name in nonmatching:
                tt = all_p[name].get("tissue_type", "unknown")
                tt_lbl = TISSUE_CATEGORIES.get(tt, {}).get("label", tt)
                rm_key = (all_p[name].get("params") or {}).get("repair_model", "monoexp")
                rm_lbl = REPAIR_MODEL_LABELS.get(rm_key, rm_key)
                cmb.addItem(f"⚠ {name}  [{tt_lbl} · {rm_lbl}]", name)

        # Restaurar selección
        if current:
            for i in range(cmb.count()):
                if cmb.itemData(i) == current:
                    cmb.setCurrentIndex(i)
                    break
        cmb.blockSignals(False)

    # ── Estado de cada fila ────────────────────────────────────────────────────

    def _refresh_state(self, entry: dict):
        preset_name = entry["cmb_preset"].currentData()
        tt_selected = entry["cmb_tissue"].currentData() or "unknown"

        if not preset_name:
            entry["lbl_state"].setText("⊘  No se calculará IsoE")
            entry["lbl_state"].setStyleSheet("color:#9E9E9E;font-style:italic;")
            return

        all_p = _all_presets()
        preset = all_p.get(preset_name, {})
        tt_preset = preset.get("tissue_type", "unknown")
        rm_key    = (preset.get("params") or {}).get("repair_model", "monoexp")
        rm_lbl    = REPAIR_MODEL_LABELS.get(rm_key, rm_key)
        valid_o   = preset.get("valid_organs", [])
        organ     = entry["organ"]

        lines = []

        # ¿El tissue_type coincide?
        if tt_preset != tt_selected and tt_preset != "unknown":
            lines.append(
                f"⚠ Preset de «{TISSUE_CATEGORIES.get(tt_preset,{}).get('label',tt_preset)}» "
                f"aplicado a tejido «{TISSUE_CATEGORIES.get(tt_selected,{}).get('label',tt_selected)}»"
            )
            color = "#B71C1C"
        elif valid_o and organ not in valid_o:
            lines.append(f"⚠ Órgano no listado en valid_organs del preset")
            color = "#E65100"
        else:
            lines.append(f"✓ Compatible")
            color = "#1B5E20"

        lines.append(f"Cinética: {rm_lbl}")
        alv = preset.get("approximation_level", "")
        alv_lbl = {"full": "Completo", "standard": "Estándar", "manual": "Manual"}.get(alv, alv)
        if alv_lbl:
            lines.append(f"Aprox.: {alv_lbl}")

        entry["lbl_state"].setText("\n".join(lines))
        entry["lbl_state"].setStyleSheet(f"color:{color};")

    def _update_summary(self):
        total = len(self._rows)
        assigned = sum(1 for e in self._rows if e["cmb_preset"].currentData())
        skipped  = total - assigned
        warn = sum(
            1 for e in self._rows
            if e["cmb_preset"].currentData() and
               _all_presets().get(e["cmb_preset"].currentData(), {}).get("tissue_type")
               != e["cmb_tissue"].currentData()
        )
        parts = [f"<b>{assigned}/{total}</b> órganos con IsoE calculado"]
        if skipped:  parts.append(f"<b>{skipped}</b> omitidos")
        if warn:     parts.append(f"<span style='color:#B71C1C'><b>{warn}</b> advertencias de tejido</span>")
        self._lbl_summary.setText("  |  ".join(parts))

    # ── Acciones ──────────────────────────────────────────────────────────────

    def _on_tissue_changed(self, entry: dict):
        tt = entry["cmb_tissue"].currentData() or "unknown"
        cur = entry["cmb_preset"].currentData()
        self._rebuild_preset_combo(entry["cmb_preset"], tt, cur)
        self._refresh_state(entry)
        self._update_summary()

    def _autodetect_all(self):
        assignment = auto_assign_presets(self._organs)
        for entry in self._rows:
            organ = entry["organ"]
            tt    = detect_tissue_type(organ)
            preset = assignment.get(organ)

            entry["cmb_tissue"].blockSignals(True)
            idx_t = entry["cmb_tissue"].findData(tt)
            entry["cmb_tissue"].setCurrentIndex(max(idx_t, 0))
            entry["cmb_tissue"].blockSignals(False)

            self._rebuild_preset_combo(entry["cmb_preset"], tt, preset)
            self._refresh_state(entry)

        self._update_summary()

    def _clear_all(self):
        for entry in self._rows:
            entry["cmb_preset"].blockSignals(True)
            entry["cmb_preset"].setCurrentIndex(0)
            entry["cmb_preset"].blockSignals(False)
            self._refresh_state(entry)
        self._update_summary()

    def _open_preset_manager(self):
        dlg = IsoEPresetsDialog(self)
        dlg.exec()
        # Reconstruir combos porque pueden haber cambiado los presets de usuario
        for entry in self._rows:
            tt  = entry["cmb_tissue"].currentData() or "unknown"
            cur = entry["cmb_preset"].currentData()
            self._rebuild_preset_combo(entry["cmb_preset"], tt, cur)
            self._refresh_state(entry)
        self._update_summary()

    # ── API pública ────────────────────────────────────────────────────────────

    def get_assignment(self) -> dict[str, str | None]:
        """Retorna {organ: preset_name | None}."""
        return {e["organ"]: e["cmb_preset"].currentData() for e in self._rows}

    def get_params_by_organ(self) -> dict[str, IsoEParams]:
        """Retorna {organ: IsoEParams} para órganos con preset asignado."""
        return build_params_by_organ(self.get_assignment(), USER_ISOE_PRESETS)

    def get_t0_maps_by_organ(self) -> dict[str, dict]:
        """Retorna {organ: t0_map} para órganos con preset asignado."""
        result = {}
        for e in self._rows:
            name = e["cmb_preset"].currentData()
            if name:
                result[e["organ"]] = get_t0_map_for_preset(name, USER_ISOE_PRESETS)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# IsoEPresetsDialog — gestor de la biblioteca de presets
# ══════════════════════════════════════════════════════════════════════════════

class IsoEPresetsDialog(QtWidgets.QDialog):
    """
    Gestor de la biblioteca de presets IsoE.
    Permite ver, crear, editar y borrar presets de usuario.
    Los presets incorporados no se pueden modificar.

    Características:
        - Scroll vertical en todo el panel de edición.
        - Filtro por categoría de tejido.
        - Nomenclatura intuitiva para los modelos de reparación.
        - Tooltips con significado físico de cada parámetro.
        - Campo de comentarios.
        - Soporte para cinética monoexp y biexp.
    """

    PARAM_KEYS = IsoEParams.NUMERIC_KEYS
    T0_KEYS    = ["Boro", "Fstn", "Thn", "Gamma", "R"]
    BIEXP_KEYS = IsoEParams.BIEXP_KEYS
    BIEXP_LABELS = [
        ("t₀f [s]  (tiempo rápido)", "Tiempo de la componente de reparación RÁPIDA [s]. Para SNC de rata: 0.7 h = 2520 s."),
        ("t₀s [s]  (tiempo lento)",  "Tiempo de la componente de reparación LENTA [s]. Para SNC de rata: 3.8 h = 13680 s."),
        ("pf  low-LET  (γ/ref)",     "Fracción de sublesiones de baja LET reparadas por la componente rápida. pf + ps = 1."),
        ("ps  low-LET  (γ/ref)",     "Fracción de sublesiones de baja LET reparadas por la componente lenta."),
        ("pf  high-LET (B/Th/Fn)",   "Fracción de sublesiones de alta LET reparadas por la componente rápida. pf + ps = 1."),
        ("ps  high-LET (B/Th/Fn)",   "Fracción de sublesiones de alta LET reparadas por la componente lenta."),
    ]

    def __init__(self, parent, current_preset_name: str = "Manual"):
        super().__init__(parent)
        self.setWindowTitle("Biblioteca de presets IsoE (MLQ)")
        self.setMinimumSize(680, 580)
        self.resize(720, 800)
        self._loading = False
        self._selected = (current_preset_name or "Manual").strip()

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        # Barra superior (fuera del scroll)
        root.addLayout(self._build_topbar())
        self._lbl_info = _lbl("", "#455A64", wrap=True)
        root.addWidget(self._lbl_info)
        self._lbl_warn = _lbl("", "#B71C1C", bold=True, wrap=True)
        self._lbl_warn.setVisible(False)
        root.addWidget(self._lbl_warn)
        root.addWidget(_separator())

        # Área con scroll
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        self._inner = QtWidgets.QVBoxLayout(inner)
        self._inner.setSpacing(10)

        self._inner.addWidget(self._build_meta_group())
        self._inner.addWidget(self._build_params_group())
        self._inner.addWidget(self._build_repair_group())
        self._inner.addLayout(self._build_ref_section())
        self._inner.addStretch()

        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Botones fijos
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # Conexiones
        self._cmb_preset.currentTextChanged.connect(self._on_preset_changed)
        self._cmb_filter.currentIndexChanged.connect(self._on_filter_changed)
        self._cmb_repair.currentTextChanged.connect(self._on_repair_changed)
        self._tbl_params.itemChanged.connect(self._on_edited)
        self._tbl_t0.itemChanged.connect(self._on_edited)
        for ed in self._biexp_edits.values():
            ed.textChanged.connect(self._on_edited)
        # Nuevos campos de metadata
        self._edit_tissue.textChanged.connect(self._on_edited)
        self._edit_model_system.textChanged.connect(self._on_edited)
        self._edit_endpoint.textChanged.connect(self._on_edited)
        self._edit_boron_compound.textChanged.connect(self._on_edited)
        self._edit_approx_notes.textChanged.connect(self._on_edited)
        self._cmb_tissue_type.currentIndexChanged.connect(
            lambda _: self._update_tissue_warn()
        )
        self._btn_save.clicked.connect(self._save)
        self._btn_delete.clicked.connect(self._delete)

        self._reload_combo()
        self._load("Manual")

    # ── Constructores de secciones ─────────────────────────────────────────────

    def _build_topbar(self) -> QtWidgets.QHBoxLayout:
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(_lbl("Filtrar:", "#455A64"))
        self._cmb_filter = QtWidgets.QComboBox()
        self._cmb_filter.addItem("Todos los tejidos", "")
        for tt, info in TISSUE_CATEGORIES.items():
            self._cmb_filter.addItem(info["label"], tt)
        bar.addWidget(self._cmb_filter)
        bar.addWidget(_lbl("  Preset:", "#455A64"))
        self._cmb_preset = QtWidgets.QComboBox()
        self._cmb_preset.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Fixed)
        bar.addWidget(self._cmb_preset, 1)
        self._btn_save   = QtWidgets.QPushButton("💾 Guardar como…")
        self._btn_delete = QtWidgets.QPushButton("🗑 Borrar")
        bar.addWidget(self._btn_save)
        bar.addWidget(self._btn_delete)
        return bar

    def _build_meta_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Información de validación del preset")
        form = QtWidgets.QFormLayout(grp)
        form.setSpacing(6)

        form.addRow(_lbl(
            "Completá estos campos al crear un preset nuevo — se guardan junto con los parámetros.",
            "#1565C0", italic=True, wrap=True,
        ))

        self._edit_tissue = QtWidgets.QLineEdit()
        self._edit_tissue.setPlaceholderText("ej: Pulmón normal (rata), Tumor gliosarcoma (9L)…")
        self._edit_tissue.setToolTip("Descripción libre del tejido al que aplican los parámetros.")
        form.addRow("Tejido:", self._edit_tissue)

        self._cmb_tissue_type = QtWidgets.QComboBox()
        for tt, info in TISSUE_CATEGORIES.items():
            self._cmb_tissue_type.addItem(info["label"], tt)
        self._cmb_tissue_type.setToolTip(
            "Categoría de tejido. Determina con qué órganos es compatible el preset."
        )
        form.addRow("Categoría:", self._cmb_tissue_type)

        self._edit_model_system = QtWidgets.QLineEdit()
        self._edit_model_system.setPlaceholderText(
            "ej: Rata Wistar, pulmón in vivo, irradiación con fotones de Co-60…"
        )
        self._edit_model_system.setToolTip(
            "Sistema experimental o clínico del que se extrajeron los parámetros."
        )
        form.addRow("Sistema modelo:", self._edit_model_system)

        self._edit_endpoint = QtWidgets.QLineEdit()
        self._edit_endpoint.setPlaceholderText(
            "ej: Supervivencia clonogénica S=0.01 · Fibrosis pulmonar LD₅₀ · Parálisis ED₅₀…"
        )
        self._edit_endpoint.setToolTip(
            "Endpoint radiobiológico medido en el que se basan los parámetros."
        )
        form.addRow("Endpoint:", self._edit_endpoint)

        self._edit_boron_compound = QtWidgets.QLineEdit()
        self._edit_boron_compound.setPlaceholderText("ej: BPA i.v. 250 mg/kg · BSH · —")
        self._edit_boron_compound.setToolTip(
            "Compuesto de boro y dosis usada en el experimento de derivación de parámetros."
        )
        form.addRow("Compuesto B:", self._edit_boron_compound)

        self._cmb_approx_level = QtWidgets.QComboBox()
        self._cmb_approx_level.addItem("Completo (sin aprox.)",              "full")
        self._cmb_approx_level.addItem("Estándar (aprox. justificadas)",     "standard")
        self._cmb_approx_level.addItem("Manual / usuario",                   "manual")
        self._cmb_approx_level.setToolTip(
            "full: todos los parámetros ajustados independientemente.\n"
            "standard: algunas aprox. documentadas (aTh=aFn, aG=aR, etc.).\n"
            "manual: preset ingresado por el usuario sin validación bibliográfica."
        )
        form.addRow("Nivel de aprox.:", self._cmb_approx_level)

        self._edit_approx_notes = QtWidgets.QPlainTextEdit()
        self._edit_approx_notes.setMaximumHeight(80)
        self._edit_approx_notes.setPlaceholderText(
            "Describí qué se aproximó y por qué (ej: aTh = aFn por LET similar; GR = 1 ref. aguda)…"
        )
        self._edit_approx_notes.setToolTip(
            "Justificación de las aproximaciones usadas. Aparece en el panel de info del preset."
        )
        form.addRow("Aproximaciones:", self._edit_approx_notes)

        return grp

    def _build_params_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Parámetros radiobiológicos (MLQ con sinergia)")
        vb = QtWidgets.QVBoxLayout(grp)
        vb.addWidget(_lbl(
            "Referencia fotónica (R): aR, bR, GR.   "
            "Componentes BNCT: aB, bB, aFn, bFn, aTh, bTh, aG, bG.   "
            "GR = escalar fijo del experimento de referencia (≠ función de θ_BNCT).",
            "#455A64", italic=True, wrap=True,
        ))
        self._tbl_params = QtWidgets.QTableWidget(len(self.PARAM_KEYS), 3)
        self._tbl_params.setHorizontalHeaderLabels(
            ["Parámetro  [unidad]", "Valor", "Significado físico"]
        )
        self._tbl_params.horizontalHeader().setStretchLastSection(True)
        self._tbl_params.verticalHeader().setVisible(False)
        self._tbl_params.setAlternatingRowColors(True)
        for i, key in enumerate(self.PARAM_KEYS):
            sym, unit, desc = PARAM_DESCRIPTIONS.get(key, (key, "", ""))
            lbl_text = f"{sym}  [{unit}]" if unit else sym
            it_k = QtWidgets.QTableWidgetItem(lbl_text)
            it_k.setFlags(it_k.flags() & ~QtCore.Qt.ItemIsEditable)
            it_k.setToolTip(desc)
            it_k.setFont(QtGui.QFont("Monospace", 9))
            self._tbl_params.setItem(i, 0, it_k)
            it_v = QtWidgets.QTableWidgetItem("0.0")
            it_v.setToolTip(desc)
            self._tbl_params.setItem(i, 1, it_v)
            short = desc[:85] + ("…" if len(desc) > 85 else "")
            it_d = QtWidgets.QTableWidgetItem(short)
            it_d.setFlags(it_d.flags() & ~QtCore.Qt.ItemIsEditable)
            it_d.setForeground(QtGui.QColor("#607D8B"))
            it_d.setToolTip(desc)
            self._tbl_params.setItem(i, 2, it_d)
        self._tbl_params.resizeColumnToContents(0)
        self._tbl_params.setColumnWidth(1, 90)
        vb.addWidget(self._tbl_params)
        return grp

    def _build_repair_group(self) -> QtWidgets.QGroupBox:
        grp = QtWidgets.QGroupBox("Modelo de reparación de sublesiones")
        vb = QtWidgets.QVBoxLayout(grp)

        row_rm = QtWidgets.QHBoxLayout()
        row_rm.addWidget(_lbl("Modelo:", "#455A64"))
        self._cmb_repair = QtWidgets.QComboBox()
        for key, label in REPAIR_MODEL_LABELS.items():
            self._cmb_repair.addItem(label, key)
        self._cmb_repair.setToolTip(
            "Único tiempo: aproximación estándar (González 2012, Apéndice I, error < 3%).\n"
            "Múltiples tiempos: cinética biexponencial LET-específica "
            "(Dattoli Viegas 2025, mayor precisión para tejido nervioso)."
        )
        row_rm.addWidget(self._cmb_repair)
        row_rm.addStretch()
        vb.addLayout(row_rm)

        self._lbl_repair_note = _lbl("", "#455A64", italic=True, wrap=True)
        vb.addWidget(self._lbl_repair_note)

        # Stack: página 0 = monoexp, página 1 = biexp
        self._stack = QtWidgets.QStackedWidget()

        # Página 0: monoexp
        pg0 = QtWidgets.QWidget()
        vb0 = QtWidgets.QVBoxLayout(pg0)
        vb0.setContentsMargins(0, 4, 0, 0)
        vb0.addWidget(_lbl(
            "Un único tiempo de reparación t₀ por componente. "
            "Nota: la fila «R» es solo informativa — GR se toma del parámetro GR.",
            "#607D8B", italic=True, wrap=True,
        ))
        self._tbl_t0 = QtWidgets.QTableWidget(len(self.T0_KEYS), 3)
        self._tbl_t0.setHorizontalHeaderLabels(["Componente", "t₀ [s]", "t₀ [h]"])
        self._tbl_t0.verticalHeader().setVisible(False)
        self._tbl_t0.setAlternatingRowColors(True)
        t0_tips = {
            "Boro":  "Tiempo de reparación de sublesiones de BORO (alta LET).",
            "Fstn":  "Tiempo de reparación de sublesiones de NEUTRONES RÁPIDOS.",
            "Thn":   "Tiempo de reparación de sublesiones de NEUTRONES TÉRMICOS.",
            "Gamma": "Tiempo de reparación de sublesiones de GAMMA del haz (baja LET).",
            "R":     "Informativo — el GR de la referencia fotónica se toma del campo GR en parámetros, NO de aquí.",
        }
        for i, key in enumerate(self.T0_KEYS):
            it_k = QtWidgets.QTableWidgetItem(key)
            it_k.setFlags(it_k.flags() & ~QtCore.Qt.ItemIsEditable)
            it_k.setToolTip(t0_tips.get(key, ""))
            self._tbl_t0.setItem(i, 0, it_k)
            it_v = QtWidgets.QTableWidgetItem("3600.0")
            it_v.setToolTip(t0_tips.get(key, ""))
            self._tbl_t0.setItem(i, 1, it_v)
            it_h = QtWidgets.QTableWidgetItem("1.000 h")
            it_h.setFlags(it_h.flags() & ~QtCore.Qt.ItemIsEditable)
            it_h.setForeground(QtGui.QColor("#90A4AE"))
            self._tbl_t0.setItem(i, 2, it_h)
        self._tbl_t0.resizeColumnToContents(0)
        self._tbl_t0.itemChanged.connect(self._update_t0h)
        vb0.addWidget(self._tbl_t0)
        self._stack.addWidget(pg0)

        # Página 1: biexp
        pg1 = QtWidgets.QWidget()
        form1 = QtWidgets.QFormLayout(pg1)
        form1.setSpacing(6)
        form1.addRow(_lbl(
            "Dos poblaciones de sublesiones con diferentes velocidades de reparación.\n"
            "Los tiempos t₀f/t₀s son LET-independientes. "
            "Las fracciones pf/ps son LET-específicas.",
            "#1A237E", italic=True, wrap=True,
        ))
        self._biexp_edits: dict[str, QtWidgets.QLineEdit] = {}
        biexp_defaults = {
            "t0f_s": "2520.0", "t0s_s": "13680.0",
            "pf_lowLET": "0.38", "ps_lowLET": "0.62",
            "pf_highLET": "0.20", "ps_highLET": "0.80",
        }
        for key, (label, tip) in zip(self.BIEXP_KEYS, self.BIEXP_LABELS):
            ed = QtWidgets.QLineEdit(biexp_defaults[key])
            ed.setMaximumWidth(120)
            ed.setToolTip(tip)
            lbl_w = QtWidgets.QLabel(label + ":")
            lbl_w.setToolTip(tip)
            self._biexp_edits[key] = ed
            form1.addRow(lbl_w, ed)
        self._stack.addWidget(pg1)
        vb.addWidget(self._stack)
        return grp

    def _build_ref_section(self) -> QtWidgets.QVBoxLayout:
        vb = QtWidgets.QVBoxLayout()
        vb.setSpacing(6)

        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(_lbl("Referencia bibliográfica:", "#455A64"))
        self._edit_ref = QtWidgets.QLineEdit()
        self._edit_ref.setPlaceholderText("Autor et al., Revista (año) — Tabla/Fig …")
        r1.addWidget(self._edit_ref, 1)
        vb.addLayout(r1)

        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(_lbl("Órganos válidos (separados por coma):", "#455A64"))
        self._edit_organs = QtWidgets.QLineEdit()
        self._edit_organs.setPlaceholderText(
            "ej: Tumor, GTV, CTV   (vacío = sin restricción de órgano)"
        )
        self._edit_organs.setToolTip(
            "Claves de órganos para los que estos parámetros tienen "
            "validación bibliográfica. Separar con comas.\n"
            "Órganos fuera de la lista se calculan con advertencia."
        )
        r2.addWidget(self._edit_organs, 1)
        vb.addLayout(r2)

        vb.addWidget(_lbl("Comentarios / notas de uso:", "#455A64"))
        self._edit_comments = QtWidgets.QPlainTextEdit()
        self._edit_comments.setMaximumHeight(75)
        self._edit_comments.setPlaceholderText(
            "Origen de los parámetros, limitaciones, poblaciones estudiadas…"
        )
        vb.addWidget(self._edit_comments)
        return vb

    # ── Combo y filtro ─────────────────────────────────────────────────────────

    def _reload_combo(self, filter_tt: str = ""):
        cur = self._selected
        self._cmb_preset.blockSignals(True)
        self._cmb_preset.clear()
        all_p = _all_presets()
        for name, preset in all_p.items():
            tt = preset.get("tissue_type", "unknown")
            if filter_tt and tt != filter_tt:
                continue
            self._cmb_preset.addItem(name)
        idx = self._cmb_preset.findText(cur)
        self._cmb_preset.setCurrentIndex(max(idx, 0))
        self._cmb_preset.blockSignals(False)

    def _on_filter_changed(self):
        self._reload_combo(self._cmb_filter.currentData() or "")
        self._load(self._cmb_preset.currentText() or "Manual")

    # ── Carga de preset ────────────────────────────────────────────────────────

    def _load(self, name: str):
        self._loading = True
        try:
            p = _all_presets().get(name, {})
            params  = p.get("params") or {}
            t0_map  = p.get("t0_map") or {}
            tt      = p.get("tissue_type", "unknown")

            self._edit_tissue.setText(p.get("tissue", ""))
            idx_tt = self._cmb_tissue_type.findData(tt)
            self._cmb_tissue_type.setCurrentIndex(max(idx_tt, 0))
            self._edit_model_system.setText(p.get("model_system", ""))
            self._edit_endpoint.setText(p.get("endpoint", ""))
            self._edit_boron_compound.setText(p.get("boron_compound", ""))
            alv = p.get("approximation_level", "manual")
            idx_alv = self._cmb_approx_level.findData(alv)
            self._cmb_approx_level.setCurrentIndex(max(idx_alv, 0))
            self._edit_approx_notes.setPlainText(p.get("approx_notes", ""))

            # Advertencia por tipo de tejido
            self._update_tissue_warn(tt)

            # Parámetros numéricos
            for i, key in enumerate(self.PARAM_KEYS):
                self._tbl_params.item(i, 1).setText(
                    f"{float(params.get(key, 0.0)):.8g}"
                )

            # Modelo de reparación
            rm = params.get("repair_model", "monoexp")
            idx_rm = self._cmb_repair.findData(rm)
            self._cmb_repair.blockSignals(True)
            self._cmb_repair.setCurrentIndex(max(idx_rm, 0))
            self._cmb_repair.blockSignals(False)
            self._switch_repair(rm)

            if rm == "biexp":
                for key in self.BIEXP_KEYS:
                    v = params.get(key, float(self._biexp_edits[key].text()))
                    self._biexp_edits[key].setText(f"{float(v):.6g}")
            else:
                for i, key in enumerate(self.T0_KEYS):
                    self._tbl_t0.item(i, 1).setText(
                        f"{float(t0_map.get(key, 3600.0)):.6g}"
                    )
                self._update_t0h()

            self._edit_ref.setText(str(p.get("ref", "")))
            self._edit_organs.setText(", ".join(p.get("valid_organs", [])))
            self._edit_comments.setPlainText(p.get("comments", ""))

        finally:
            self._loading = False

        self._selected = name
        alv2 = _all_presets().get(name, {}).get("approximation_level", "")
        alv2_lbl = {"full": "cálculo completo",
                    "standard": "aprox. estándar",
                    "manual": "manual"}.get(alv2, alv2)
        self._lbl_info.setText(
            f"Preset: <b>{name}</b>  |  {alv2_lbl}"
            if name != "Manual" else
            "Modo manual — editá los parámetros directamente."
        )

    def _update_tissue_warn(self, tt: str | None = None):
        """Actualiza el banner de advertencia según el tissue_type seleccionado."""
        if tt is None:
            tt = self._cmb_tissue_type.currentData() or "unknown"
        if tt == "tumor":
            self._lbl_warn.setText(
                "⚠ PARÁMETROS TUMORALES — No aplicar a órganos de tejido sano."
            )
            self._lbl_warn.setVisible(True)
        elif tt in ("normal_brain", "spinal_cord", "skin", "mucosa"):
            self._lbl_warn.setText(
                "⚠ PARÁMETROS DE TEJIDO SANO — No aplicar a volúmenes tumorales."
            )
            self._lbl_warn.setVisible(True)
        else:
            self._lbl_warn.setVisible(False)

    def _switch_repair(self, rm_key: str):
        if rm_key == "biexp":
            self._stack.setCurrentIndex(1)
            self._lbl_repair_note.setText(
                "Múltiples tiempos de reparación activos. "
                "Tiempos LET-independientes; fracciones LET-específicas. "
                "Referencia: Dattoli Viegas et al. (2025)."
            )
            self._lbl_repair_note.setStyleSheet("color:#1A237E;font-style:italic;")
        else:
            self._stack.setCurrentIndex(0)
            self._lbl_repair_note.setText(
                "Único tiempo de reparación por componente. "
                "Approx. válida para tiempos BNCT 10–30 min "
                "(González 2012, Apéndice I, error < 3%)."
            )
            self._lbl_repair_note.setStyleSheet("color:#455A64;font-style:italic;")

    def _update_t0h(self):
        if self._loading:
            return
        self._tbl_t0.blockSignals(True)
        for i in range(self._tbl_t0.rowCount()):
            it = self._tbl_t0.item(i, 1)
            it_h = self._tbl_t0.item(i, 2)
            if it and it_h:
                try:
                    it_h.setText(f"{float(it.text().replace(',','.'))/3600:.3f} h")
                except ValueError:
                    it_h.setText("—")
        self._tbl_t0.blockSignals(False)

    # ── Señales ────────────────────────────────────────────────────────────────

    def _on_preset_changed(self, text: str):
        text = (text or "").strip()
        if text:
            self._load(text)

    def _on_repair_changed(self, _):
        if self._loading:
            return
        rm = self._cmb_repair.currentData() or "monoexp"
        self._switch_repair(rm)
        if _is_builtin_isoe_preset(self._selected):
            self._lbl_info.setText(
                f"Editando sobre «{self._selected}» — usar «Guardar como…» para crear un preset nuevo."
            )

    def _on_edited(self, *_):
        if self._loading:
            return
        if _is_builtin_isoe_preset(self._selected):
            self._lbl_info.setText(
                f"Editando sobre «{self._selected}» — usar «Guardar como…» para crear un preset nuevo."
            )

    # ── Leer formulario ────────────────────────────────────────────────────────

    def _read_form(self):
        params = {}
        for i, key in enumerate(self.PARAM_KEYS):
            it = self._tbl_params.item(i, 1)
            txt = (it.text() if it else "0").strip().replace(",", ".")
            try:
                params[key] = float(txt)
            except ValueError:
                raise ValueError(f"Valor inválido para {key}: '{txt}'")

        rm = self._cmb_repair.currentData() or "monoexp"
        params["repair_model"] = rm
        t0_map = {}

        if rm == "biexp":
            for key in self.BIEXP_KEYS:
                txt = self._biexp_edits[key].text().strip().replace(",", ".")
                try:
                    v = float(txt)
                except ValueError:
                    raise ValueError(f"Valor inválido para {key}: '{txt}'")
                if key in ("t0f_s", "t0s_s") and v <= 0:
                    raise ValueError(f"{key} debe ser > 0.")
                if key.startswith(("pf_", "ps_")) and not (0 <= v <= 1):
                    raise ValueError(f"{key} debe estar en [0,1].")
                params[key] = v
            for pf, ps in [("pf_lowLET", "ps_lowLET"), ("pf_highLET", "ps_highLET")]:
                s = params[pf] + params[ps]
                if abs(s - 1.0) > 0.01:
                    raise ValueError(f"{pf} + {ps} = {s:.4f} ≠ 1 (deben sumar 1 ± 0.01).")
        else:
            for i, key in enumerate(self.T0_KEYS):
                it = self._tbl_t0.item(i, 1)
                txt = (it.text() if it else "3600").strip().replace(",", ".")
                try:
                    v = float(txt)
                    if v <= 0:
                        raise ValueError(f"t0 debe ser > 0 para {key}.")
                    t0_map[key] = v
                except ValueError as e:
                    raise ValueError(f"t0[{key}]: {e}")

        ref           = self._edit_ref.text().strip()
        organs        = [x.strip() for x in self._edit_organs.text().split(",") if x.strip()]
        comments      = self._edit_comments.toPlainText().strip()
        tissue        = self._edit_tissue.text().strip()
        tissue_type   = self._cmb_tissue_type.currentData() or "unknown"
        model_system  = self._edit_model_system.text().strip()
        endpoint      = self._edit_endpoint.text().strip()
        boron_compound = self._edit_boron_compound.text().strip()
        approx_level  = self._cmb_approx_level.currentData() or "manual"
        approx_notes  = self._edit_approx_notes.toPlainText().strip()
        return (params, t0_map, ref, rm, organs, comments,
                tissue, tissue_type, model_system, endpoint,
                boron_compound, approx_level, approx_notes)

    # ── Guardar / borrar ───────────────────────────────────────────────────────

    def _save(self):
        try:
            (params, t0_map, ref, rm, organs, comments,
             tissue, tissue_type, model_system, endpoint,
             boron_compound, approx_level, approx_notes) = self._read_form()
            IsoEParams(**{k: v for k, v in params.items()
                          if k in IsoEParams.NUMERIC_KEYS + IsoEParams.BIEXP_KEYS + ["repair_model"]})
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "IsoE — Error", f"Parámetros inválidos:\n{e}")
            return

        name, ok = QtWidgets.QInputDialog.getText(
            self, "Guardar preset IsoE", "Nombre del nuevo preset:"
        )
        name = (name or "").strip()
        if not ok or not name:
            return
        if _is_builtin_isoe_preset(name):
            QtWidgets.QMessageBox.warning(
                self, "IsoE", f"No se puede sobrescribir «{name}» (preset incorporado)."
            )
            return
        if name in ISOE_PARAM_PRESETS:
            if QtWidgets.QMessageBox.question(
                self, "Sobrescribir", f"Ya existe «{name}». ¿Sobrescribirlo?"
            ) != QtWidgets.QMessageBox.Yes:
                return

        new_preset = {
            "ref":                 ref or "Usuario",
            "tissue":              tissue or "Sin especificar (usuario)",
            "tissue_type":         tissue_type,
            "valid_organs":        organs,
            "model_system":        model_system or "Ingresado por el usuario",
            "endpoint":            endpoint or "—",
            "boron_compound":      boron_compound or "—",
            "approximation_level": approx_level,
            "approx_notes":        approx_notes or "Preset ingresado manualmente.",
            "comments":            comments,
            "params":              params,
            "t0_map":              t0_map if rm == "monoexp" else None,
        }
        ISOE_PARAM_PRESETS[name] = new_preset
        USER_ISOE_PRESETS[name]  = new_preset
        ok_disk, err = save_user_isoe_presets(USER_ISOE_PRESETS)
        if not ok_disk:
            QtWidgets.QMessageBox.warning(self, "IsoE", f"No se pudo guardar en disco:\n{err}")
        self._selected = name
        self._reload_combo(self._cmb_filter.currentData() or "")
        self._cmb_preset.blockSignals(True)
        self._cmb_preset.setCurrentText(name)
        self._cmb_preset.blockSignals(False)
        self._lbl_info.setText(f"Preset guardado: <b>{name}</b>")
        self._lbl_warn.setVisible(False)

    def _delete(self):
        name = self._cmb_preset.currentText().strip()
        if _is_builtin_isoe_preset(name):
            QtWidgets.QMessageBox.warning(self, "IsoE", "No se puede borrar un preset incorporado.")
            return
        if QtWidgets.QMessageBox.question(
            self, "Borrar", f"¿Borrar «{name}»? No se puede deshacer."
        ) != QtWidgets.QMessageBox.Yes:
            return
        ISOE_PARAM_PRESETS.pop(name, None)
        USER_ISOE_PRESETS.pop(name, None)
        ok_disk, err = save_user_isoe_presets(USER_ISOE_PRESETS)
        if not ok_disk:
            QtWidgets.QMessageBox.warning(self, "IsoE", f"No se pudo guardar en disco:\n{err}")
        self._selected = "Manual"
        self._reload_combo(self._cmb_filter.currentData() or "")
        self._cmb_preset.blockSignals(True)
        self._cmb_preset.setCurrentText("Manual")
        self._cmb_preset.blockSignals(False)
        self._load("Manual")
        self._lbl_info.setText("Preset borrado.")

    def selected_preset_name(self) -> str:
        return self._selected


# ══════════════════════════════════════════════════════════════════════════════
# IsoEDialog — editor rápido (acceso legacy)
# ══════════════════════════════════════════════════════════════════════════════

class IsoEDialog(QtWidgets.QDialog):
    """Editor rápido de parámetros numéricos. Para gestión completa usar IsoEPresetsDialog."""

    PARAM_KEYS = IsoEParams.NUMERIC_KEYS

    def __init__(self, parent, current_params: dict):
        super().__init__(parent)
        self.setWindowTitle("Parámetros IsoE (MLQ) — Editor rápido")
        self.setMinimumWidth(540)
        self.params = dict(current_params) if isinstance(current_params, dict) else {}

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)
        root.addWidget(_lbl(
            "ℹ GR = factor de Lea-Catcheside de la irradiación de REFERENCIA fotónica "
            "(escalar fijo, NO función de θ_BNCT). Usar GR = 1.0 para referencia aguda.",
            "#1565C0", italic=True, wrap=True,
        ))
        self._tbl = QtWidgets.QTableWidget(len(self.PARAM_KEYS), 3)
        self._tbl.setHorizontalHeaderLabels(["Parámetro [unidad]", "Valor", "Significado"])
        self._tbl.horizontalHeader().setStretchLastSection(True)
        self._tbl.verticalHeader().setVisible(False)
        for i, key in enumerate(self.PARAM_KEYS):
            sym, unit, desc = PARAM_DESCRIPTIONS.get(key, (key, "", ""))
            lbl_text = f"{sym}  [{unit}]" if unit else sym
            it_k = QtWidgets.QTableWidgetItem(lbl_text)
            it_k.setFlags(it_k.flags() & ~QtCore.Qt.ItemIsEditable)
            it_k.setToolTip(desc)
            self._tbl.setItem(i, 0, it_k)
            self._tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(str(self.params.get(key, 0.0))))
            short = desc[:75] + ("…" if len(desc) > 75 else "")
            it_d = QtWidgets.QTableWidgetItem(short)
            it_d.setFlags(it_d.flags() & ~QtCore.Qt.ItemIsEditable)
            it_d.setForeground(QtGui.QColor("#90A4AE"))
            self._tbl.setItem(i, 2, it_d)
        self._tbl.resizeColumnToContents(0)
        self._tbl.setColumnWidth(1, 90)
        root.addWidget(self._tbl)
        rm = self.params.get("repair_model", "monoexp")
        root.addWidget(_lbl(
            f"Modelo de reparación: {REPAIR_MODEL_LABELS.get(rm, rm)}. "
            "Para editar parámetros biexponenciales usar «Gestionar presets IsoE».",
            "#455A64", italic=True, wrap=True,
        ))
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def get_params(self) -> dict:
        out = {}
        for i, key in enumerate(self.PARAM_KEYS):
            it = self._tbl.item(i, 1)
            try:
                out[key] = float(it.text().replace(",", ".")) if it else 0.0
            except ValueError:
                out[key] = 0.0
        return out


# ══════════════════════════════════════════════════════════════════════════════
# SPND registry (sin cambios)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SPND_REGISTRY = {
    "version": 3,
    "updated_label": "Actualizados a may 2024",
    "units": {"current": "pA", "sensitivity": "A/(n/cm^2/s)"},
    "detectors": [
        {"name": "Rojo",        "factor_to_verde": 1.02541600, "sens": 1.90e-21, "sens_sigma": 1.5e-22, "last_calib": "", "notes": ""},
        {"name": "A0",          "factor_to_verde": 0.98599951, "sens": None,     "sens_sigma": None,     "last_calib": "", "notes": ""},
        {"name": "A1",          "factor_to_verde": 1.00603069, "sens": None,     "sens_sigma": None,     "last_calib": "", "notes": ""},
        {"name": "Verde (ref)", "factor_to_verde": 1.0,        "sens": 1.98e-21, "sens_sigma": 1.6e-22, "last_calib": "", "notes": ""},
    ],
}
