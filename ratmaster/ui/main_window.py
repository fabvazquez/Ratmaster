"""
ui/main_window.py
=================
Ventana principal de RatMaster (BNCTMain).

Responsabilidades:
  - Layout principal: panel de parámetros (izquierda) + DVH + lista de órganos.
  - Carga de vectores de dosis desde la carpeta activa.
  - Ejecución del cálculo dosimétrico (compute_bnct + compute_isoe_from_report).
  - Gestión del gráfico DVH: modelo de checkboxes "sin check = todos visibles".
  - Menús: configuración de SPND, protocolos de boro, constraints, presets IsoE.
  - Exportación de resultados (ResultsDialog).

El cálculo pesado se delega a physics/compute_bnct.py y physics/isoe.py.
"""

import json
import shutil
from datetime import datetime
import numpy as np
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from ratmaster.app_paths import (
    app_install_dir, user_app_dir, ensure_user_json,
    bundled_icon_path, ensure_user_vectors,
)
from ratmaster.constants import (
    ORG_ORDER, FLUX_CORR_FACTOR, ORGAN_COLORS,
    PROTO_LIB, BIO_LIB, DEFAULT_BIO_PRESET_NAME,
    CONSTRAINT_PRESETS,
    USER_BORO_PROTOCOLS, USER_CONSTRAINT_PRESETS, USER_ISOE_PRESETS, USER_BIO_PRESETS,
    _defaults_from_libs, _make_constraints_matrix,
    _resolve_boro_protocol_name, _is_builtin_boro_protocol,
    _is_builtin_isoe_preset, _is_builtin_constraint_preset, _is_builtin_bio_preset,
    _constraints_matrix_from_serializable, _sanitize_boro_protocol_dict,
    _constraints_matrix_to_dict,
)
from ratmaster.data.vector_loader import load_vectordose, _safe_set_name
from ratmaster.data.persistence import load_spnd_registry, save_spnd_registry, parse_number_or_pair
from ratmaster.physics.dose_utils import (
    pad_to_N, build_dvh, dvh_extend_to_zero,
    metrics_with_uncertainty, numpy_to_list, achieved_value_for_constraint,
    summarize_constraints_matrix,  # movido de constants → dose_utils donde está definido
)
from ratmaster.physics.compute_bnct import compute_bnct
from ratmaster.physics.isoe import (
    IsoEParams, ISOE_PARAM_PRESETS, DEFAULT_T0_MAP,
    compute_isoe_from_report, solve_time_isoe_from_constraints,
    auto_assign_presets, build_params_by_organ, detect_tissue_type,
)
from ratmaster.ui.formatters import (
    format_value_uncertainty, format_scientific_value_uncertainty, format_time, parse_time_input,
    dose_type_display, dose_axis_unit,
)
from ratmaster.ui.canvas import MplCanvas
from ratmaster.ui.dialogs.results import ResultsDialog, ComponentDosesDialog, build_viz_data
from ratmaster.ui.dialogs.bio_params import BioParamsDialog
from ratmaster.ui.dialogs.boro import BoroDialog
from ratmaster.ui.dialogs.isoe_dialogs import IsoEDialog, IsoEPresetsDialog, IsoEOrganParamsDialog
from ratmaster.ui.dialogs.constraints import ConstraintsDialog
from ratmaster.ui.dialogs.spnd import SPNDConfigDialog, SPNDFromCurrentsDialog
from ratmaster.ui.dialogs.vector_gen import VectorGenDialog


class BNCTMain(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ratmaster")
        self.setMinimumSize(980, 600)
        try:
            icon_path = bundled_icon_path()
            if icon_path is not None:
                self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        except Exception:
            pass

        # carpetas de instalación y de datos del usuario
        self.install_dir = app_install_dir()
        self.base_folder = user_app_dir()
        ensure_user_vectors()
        # --- Vector sets ---
        self.vectors_root = (self.base_folder / "Vectores").resolve()
        self.vectors_root.mkdir(parents=True, exist_ok=True)
        self.config_path = ensure_user_json("ratmaster_config.json", {
            "active_vector_set": "DEFAULT",
            "custom_boro_protocols": {},
            "custom_constraint_presets": {},
            "custom_isoe_presets": {}
        })
        self.active_vector_set = "DEFAULT"
        self._load_config()
        self._ensure_default_vector_set()
        self._set_active_vector_set(self.active_vector_set, persist=False)

        # defaults para el generador SEG+MESH
        self.last_seg_path = ""
        self.last_meshtal_path = ""
        self.last_voxel_size_mm = (0.78125, 0.3325, 0.3325)
        self.last_origin_mesh_cm = (230.0, -2.25, -5.0)
        self.last_factor_spnd = 3.6006e-9
        self.last_tallies = [14, 24, 34, 44]
        self.last_tally_overlay = 14

        # data
        self.vectors = {}   # organ -> (B,F,T,G)
        self.B_arr = None; self.B_err = None; self.CBE = None; self.RBE = None; self.Constraints = None
        self.active_boro_protocol_name = "BPA46.5"
        # Preset de parámetros biológicos (CBE/RBE) activo — ver BIO_LIB en
        # constants.py y ui/dialogs/bio_params.py.
        self.active_bio_preset_name = DEFAULT_BIO_PRESET_NAME
        # Origen de la concentración de boro: "generico" (valores del protocolo
        # tal cual) o "sangre" (calculada a partir de una medición en sangre y
        # la relación Tejido/Sangre del protocolo). Ver ui/dialogs/boro.py.
        self.boro_source_mode = "generico"
        self.boro_blood_conc = None
        self.boro_blood_conc_err = None

        # resultados
        self.report = None; self.dsum = None; self.deq = None
        # viz_data: datos espaciales para el visor 2D de dosis (DoseViewerDialog).
        # Se llena en open_vector_gen_dialog() si los vectores se generaron en esta sesión,
        # o puede cargarse desde disco. Si es None, el botón "Visualizar Dosis" queda
        # deshabilitado en ResultsDialog.
        self.viz_data: dict | None = None
        # Cache de DVH: {organ_key: (s_array, vol_array)} precalculados.
        # Se invalida completamente al inicio de run_calc() para que un
        # nuevo cálculo NUNCA muestre datos del anterior.
        self._dvh_cache: dict = {}
        # ID del reporte activo: se usa para detectar si el cache es stale.
        self._dvh_cache_report_id: int = 0
        self.use_bio_mode = False  # si True: wGy para DVH/métricas/constraints
        self.use_isoe_mode = False  # si True: mostrar IsoE (MLQ)
        self.isoe_params_manual = None  # parámetros manuales IsoE definidos por el usuario
        # Asignación de preset por órgano para IsoE: {organ_key: preset_name | None}
        # Se configura en IsoEOrganParamsDialog; None = no calcular IsoE para ese órgano
        self.isoe_organ_assignment: dict = {}
        # Snapshot del último cálculo de flujo desde corrientes (para rellenar el diálogo al reabrirlo y para el PDF)
        self.last_spnd_currents_snapshot: dict | None = None

        self._build_ui()
        self.showMaximized()
        # actualizar label de set una vez construida la UI
        self._set_active_vector_set(self.active_vector_set, persist=False)
        self.load_defaults()
        # SPND: asistente desde corrientes (planilla)
        if hasattr(self, "btn_spnd_from_curr"):
            self.btn_spnd_from_curr.clicked.connect(self.open_spnd_from_currents)
        # --- Menú tipo "hamburger" ---
        menu = self.menuBar()
        main_menu = menu.addMenu("≡")

        # Acción: Parámetros biológicos
        act_bio = QtGui.QAction("Parámetros biológicos (CBE / RBE)", self)
        act_bio.triggered.connect(self.open_bio_dialog)
        main_menu.addAction(act_bio)

        act_boro = QtGui.QAction("Gestionar protocolos de Boro…", self)
        act_boro.triggered.connect(self.open_boro_dialog)
        main_menu.addAction(act_boro)

        act_isoe_presets = QtGui.QAction("Gestionar presets IsoE (MLQ)…", self)
        act_isoe_presets.triggered.connect(self.open_isoe_presets_dialog)
        main_menu.addAction(act_isoe_presets)

        main_menu.addSeparator()

        # Acción: Generar vectores (SEG + MCNP)
        act_gen = QtGui.QAction("Generar vectores de dosis…", self)
        act_gen.triggered.connect(self.open_vector_gen_dialog)
        main_menu.addAction(act_gen)

        # Vector sets: elegir cuál usar
        act_set = QtGui.QAction("Cambiar set de vectores…", self)
        act_set.triggered.connect(self.open_change_vector_set_dialog)
        main_menu.addAction(act_set)

        act_open_set = QtGui.QAction("Abrir carpeta del set activo", self)
        act_open_set.triggered.connect(self.open_active_vectors_folder)
        main_menu.addAction(act_open_set)

        act_del_set = QtGui.QAction("Borrar set de vectores…", self)
        act_del_set.triggered.connect(self.open_delete_vector_set_dialog)
        main_menu.addAction(act_del_set)

        main_menu.addSeparator()

        act_spnd = QtGui.QAction("SPND: flujo desde corrientes…", self)
        act_spnd.triggered.connect(self.open_spnd_from_currents)
        main_menu.addAction(act_spnd)

        act_spnd_cfg = QtGui.QAction("SPND: configurar/calibración…", self)
        act_spnd_cfg.triggered.connect(self.open_spnd_config)
        main_menu.addAction(act_spnd_cfg)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        # PANEL IZQUIERDO Y DERECHO COMO ANTES
        self.lbl_vectors_set = QtWidgets.QLabel("Vectores: (no cargados)")
        self.lbl_vectors_set.setStyleSheet("color:#455A64; font-weight:700; font-size:9.5pt; letter-spacing:0.3px;")
        main.addWidget(self.lbl_vectors_set)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(splitter)

        # ------------------------------------
        # PANEL IZQUIERDO (TODO ORDENADO)
        # ------------------------------------
        left = QtWidgets.QWidget()
        leftlay = QtWidgets.QVBoxLayout(left)

        # ========== Sección 1: Modo de cálculo ==========
        box_modo = QtWidgets.QGroupBox("Modo de cálculo")
        lay_modo = QtWidgets.QVBoxLayout(box_modo)

        self.rb_time = QtWidgets.QRadioButton("Tiempo fijo")
        self.rb_cons = QtWidgets.QRadioButton("Restricciones de dosis")
        self.rb_cons.setChecked(True)

        lay_modo.addWidget(self.rb_time)
        lay_modo.addWidget(self.rb_cons)
        leftlay.addWidget(box_modo)

        # ========== Sección 2: Tipo de dosis ==========
        box_tipo = QtWidgets.QGroupBox("Tipo de dosis")
        lay_tipo = QtWidgets.QVBoxLayout(box_tipo)

        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.addItems(["Dosis Física: Gy", "Dosis Equivalente pesada por RBE: Gy(RBE)", "Dosis Isoefectiva (Photon IsoE): Gy(IsoE)"])
        lay_tipo.addWidget(self.combo_type)
        leftlay.addWidget(box_tipo)


        # === Sección de parámetros IsoE (MLQ) ===
        self.box_isoe = QtWidgets.QGroupBox("Parámetros IsoE (MLQ)")
        lay_iso = QtWidgets.QVBoxLayout(self.box_isoe)

        # Etiqueta informativa: qué preset está asignado a cada órgano
        self.lbl_isoe_status = QtWidgets.QLabel("Sin configurar — presioná el botón para asignar parámetros por órgano.")
        self.lbl_isoe_status.setWordWrap(True)
        self.lbl_isoe_status.setStyleSheet("color:#546E7A; font-size:9pt;")
        lay_iso.addWidget(self.lbl_isoe_status)

        # Botón principal: abre IsoEOrganParamsDialog
        self.btn_edit_isoe = QtWidgets.QPushButton("Configurar parámetros por órgano…")
        self.btn_edit_isoe.setToolTip(
            "Abre el configurador donde podés asignar parámetros radiobiológicos\n"
            "diferentes para cada órgano según su tipo de tejido.\n"
            "El sistema auto-detecta el tejido y sugiere el preset compatible."
        )
        lay_iso.addWidget(self.btn_edit_isoe)
        self.btn_edit_isoe.clicked.connect(self.open_isoe_organ_dialog)

        # Agregar grupo al panel izquierdo
        leftlay.addWidget(self.box_isoe)

        # Ocultar al inicio
        self.box_isoe.hide()

        # Mostrar solo cuando el tipo de dosis es IsoE
        def update_isoe_visibility():
            if self.combo_type.currentIndex() == 2:  # 2 = Gy_isoE
                self.box_isoe.show()
            else:
                self.box_isoe.hide()

        self.combo_type.currentIndexChanged.connect(update_isoe_visibility)


        # ========== Sección 3: Protocolo de Boro ==========
        box_boro = QtWidgets.QGroupBox("Protocolo de Boro")
        lay_boro = QtWidgets.QVBoxLayout(box_boro)

        # Creamos la tabla interna de boro (no visible)
        self.tbl_boro = QtWidgets.QTableWidget(2, len(ORG_ORDER))
        self.tbl_boro.setVerticalHeaderLabels(["[B] (ppm)", "Error [B] (ppm)"])
        self.tbl_boro.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_boro.hide()

        # ========== CREAR TABLAS INTERNAS DE CBE Y RBE (OCULTAS) ==========
        self.tbl_cbe = QtWidgets.QTableWidget(1, len(ORG_ORDER))
        self.tbl_cbe.setVerticalHeaderLabels(["CBE"])
        self.tbl_cbe.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_cbe.hide()

        self.tbl_rbe = QtWidgets.QTableWidget(1, len(ORG_ORDER))
        self.tbl_rbe.setVerticalHeaderLabels(["RBE"])
        self.tbl_rbe.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_rbe.hide()


        self.combo_proto = QtWidgets.QComboBox()
        self._refresh_proto_combo()
        self.combo_proto.currentTextChanged.connect(self.apply_selected_protocol)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.combo_proto)
        lay_boro.addLayout(row)
        self.lbl_proto_preset = QtWidgets.QLabel("Protocolo activo: no seleccionado")
        self.lbl_proto_preset.setStyleSheet("color:#546E7A; font-size:9pt;")
        lay_boro.addWidget(self.lbl_proto_preset)

        leftlay.addWidget(box_boro)


        # ========== Sección 5: Parámetros generales ==========
        box_params = QtWidgets.QGroupBox("Flujo neutrónico")
        lay_par = QtWidgets.QFormLayout(box_params)

        self.input_spnd = QtWidgets.QLineEdit("3.4e9")
        self.input_spnd_err = QtWidgets.QLineEdit("2e8")
        self.input_sys = QtWidgets.QLineEdit("0.05")

        row_spnd = QtWidgets.QHBoxLayout()
        row_spnd.setContentsMargins(0,0,0,0)
        row_spnd.addWidget(self.input_spnd)
        self.btn_spnd_from_curr = QtWidgets.QPushButton("Desde corrientes…")
        self.btn_spnd_from_curr.setToolTip("Abrir asistente para calcular flujo desde corrientes SPND.")
        row_spnd.addWidget(self.btn_spnd_from_curr)
        w_spnd = QtWidgets.QWidget()
        w_spnd.setLayout(row_spnd)
        lay_par.addRow("Flujo neutrónico (n/cm²·s):", w_spnd)
        lay_par.addRow("Incertidumbre (n/cm²·s):", self.input_spnd_err)

        leftlay.addWidget(box_params)

        # ========== Sección 6: Parámetros según modo ==========
        # TIEMPO FIJO
        self.box_time = QtWidgets.QGroupBox("Tiempo fijo")
        lay_tfix = QtWidgets.QFormLayout(self.box_time)

        self.input_time = QtWidgets.QLineEdit("0.0")
        self.input_time_err = QtWidgets.QLineEdit("0.0")
        self._time_input_fmt = "s"   # "s" o "ms" — controla cómo se interpreta input_time.text()

        self.cmb_time_input_fmt = QtWidgets.QComboBox()
        self.cmb_time_input_fmt.addItem("segundos", "s")
        self.cmb_time_input_fmt.addItem("min + seg", "ms")
        self.cmb_time_input_fmt.setToolTip(
            "Formato para ingresar el tiempo: segundos puros (ej. 612) "
            "o minutos y segundos (ej. 10m 12s)"
        )

        time_input_row = QtWidgets.QWidget()
        time_input_row_lay = QtWidgets.QHBoxLayout(time_input_row)
        time_input_row_lay.setContentsMargins(0, 0, 0, 0)
        time_input_row_lay.addWidget(self.input_time)
        time_input_row_lay.addWidget(self.cmb_time_input_fmt)

        self.lbl_time_row = QtWidgets.QLabel("Tiempo (s):")

        def _on_time_input_fmt_changed():
            self._time_input_fmt = self.cmb_time_input_fmt.currentData()
            if self._time_input_fmt == "ms":
                self.lbl_time_row.setText("Tiempo (min+seg):")
                self.input_time.setPlaceholderText("ej: 10m 12s")
            else:
                self.lbl_time_row.setText("Tiempo (s):")
                self.input_time.setPlaceholderText("ej: 612")

        self.cmb_time_input_fmt.currentIndexChanged.connect(_on_time_input_fmt_changed)

        lay_tfix.addRow(self.lbl_time_row, time_input_row)
        lay_tfix.addRow("Incertidumbre Tiempo (s):", self.input_time_err)

        leftlay.addWidget(self.box_time)

        # RESTRICCIONES DE DOSIS
        self.box_cons = QtWidgets.QGroupBox("Restricciones de dosis")
        lay_cons = QtWidgets.QVBoxLayout(self.box_cons)

        # Tabla interna (se edita en popup)
        self.tbl_cons = QtWidgets.QTableWidget(5, len(ORG_ORDER))
        self.tbl_cons.setVerticalHeaderLabels(["Dosis máxima","Dosis media","Dosis mínima","% del volumen más caliente","Dosis en ese % (Dx)"])
        self.tbl_cons.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_cons.hide()
        self.tbl_cons.itemChanged.connect(self._mark_constraints_manual)

        rowc = QtWidgets.QHBoxLayout()
        rowc.setContentsMargins(6, 8, 6, 6)
        rowc.setSpacing(10)
        self.lbl_cons_preset = QtWidgets.QLabel("Restricción: no se seleccionó")
        self.lbl_cons_preset.setStyleSheet("color:#546E7A; font-size:9pt; padding-top:2px; padding-bottom:2px;")
        self.lbl_cons_preset.setMinimumHeight(24)
        rowc.addWidget(self.lbl_cons_preset, 1)

        self.btn_edit_cons = QtWidgets.QPushButton("Editar restricciones…")
        self.btn_edit_cons.setMinimumHeight(28)
        self.btn_edit_cons.clicked.connect(self.open_constraints_dialog)
        rowc.addWidget(self.btn_edit_cons, 0, QtCore.Qt.AlignVCenter)

        lay_cons.addLayout(rowc)

        hint = QtWidgets.QLabel("Se editan en un popup (incluye presets). Los valores 0 actuan como “sin restricción”.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#78909C; font-size:9pt;")
        lay_cons.addWidget(hint)

        leftlay.addWidget(self.box_cons)


        # VISIBILIDAD INICIAL
        self.box_cons.show()
        self.box_time.hide()

        # Cambiar visibilidad dinámicamente
        self.rb_time.toggled.connect(lambda: self.box_time.setVisible(self.rb_time.isChecked()))
        self.rb_time.toggled.connect(lambda: self.box_cons.setVisible(self.rb_cons.isChecked()))
        self.rb_cons.toggled.connect(lambda: self.box_cons.setVisible(self.rb_cons.isChecked()))
        self.rb_cons.toggled.connect(lambda: self.box_time.setVisible(self.rb_time.isChecked()))

        # Mensaje de estado (status)  – se mueve a la barra inferior fija
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#546E7A; font-size:9pt;")

        # ========== Botón calcular – también va a la barra inferior ==========
        btn_run = QtWidgets.QPushButton("CALCULAR")
        btn_run.setStyleSheet(
            "font-weight: 700; font-size: 11pt; padding: 12px; letter-spacing: 0.5px;"
            "background: #2E7D32; color: white; border-radius: 5px;"
            "border-bottom: 3px solid #1B5E20;"
        )
        btn_run.clicked.connect(self.run_calc)

        leftlay.addStretch(0)

        # ── Scroll area: SOLO los settings (sin status ni botón) ──────────
        scroll_left = QtWidgets.QScrollArea()
        scroll_left.setWidget(left)
        scroll_left.setWidgetResizable(True)
        scroll_left.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_left.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_left.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_left.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        # ── Barra inferior FIJA: status + CALCULAR ────────────────────────
        bottom_bar = QtWidgets.QWidget()
        bottom_bar.setStyleSheet(
            "QWidget { background: #E8ECEE; border-top: 1px solid #CFD8DC; }"
        )
        bottom_lay = QtWidgets.QVBoxLayout(bottom_bar)
        bottom_lay.setContentsMargins(10, 8, 10, 10)
        bottom_lay.setSpacing(6)
        bottom_lay.addWidget(self.status)
        bottom_lay.addWidget(btn_run)

        # ── Contenedor izquierdo: scroll arriba + barra fija abajo ────────
        left_container = QtWidgets.QWidget()
        left_container.setMinimumWidth(360)
        left_container.setMaximumWidth(520)
        lc_lay = QtWidgets.QVBoxLayout(left_container)
        lc_lay.setContentsMargins(0, 0, 0, 0)
        lc_lay.setSpacing(0)
        lc_lay.addWidget(scroll_left, 1)   # se expande
        lc_lay.addWidget(bottom_bar, 0)    # fijo

        splitter.addWidget(left_container)

        # PANEL DERECHO (DVH)
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)

        self.canvas = MplCanvas(self, width=6, height=4)
        rlay.addWidget(self.canvas)

        self.btn_show_all = QtWidgets.QPushButton("Mostrar todos los DVH")
        self.btn_show_all.setToolTip(
            "Desmarca todos los órganos y muestra todos los DVH.\n"
            "Para ver un órgano solo, marcá su checkbox."
        )
        self.btn_show_all.clicked.connect(self.show_all_dvh)
        rlay.addWidget(self.btn_show_all)

        self.org_list = QtWidgets.QListWidget()
        self.org_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.org_list.setSpacing(2)
        self.org_list.setStyleSheet("""
QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #CFD8DC;
    border-radius: 4px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 5px 8px;
    border-radius: 4px;
    color: #546E7A;
    background-color: #FFFFFF;
    border: 1px solid transparent;
}
QListWidget::item:hover {
    background-color: #F5F7FA;
    border-color: #CFD8DC;
}
/* Indicador checkbox — sin marcar: blanco con borde gris */
QListWidget::indicator {
    width: 14px;
    height: 14px;
    border: 2px solid #90A4AE;
    border-radius: 3px;
    background-color: #FFFFFF;
    margin-right: 4px;
}
/* Indicador checkbox — marcado: relleno sólido con acento */
QListWidget::indicator:checked {
    background-color: #455A64;
    border-color: #263238;
}
""")
        # Cada órgano tiene un checkbox; marcar/desmarcar actualiza el DVH
        self.org_list.itemChanged.connect(self._on_organ_check_changed)
        # Visible desde el inicio, pero no interactuable hasta que haya resultados
        self.org_list.setEnabled(True)
        self.org_list.setFocusPolicy(QtCore.Qt.NoFocus)  # opcional: evita selección visual
        rlay.addWidget(self.org_list)

        splitter.addWidget(right)

    def open_bio_dialog(self):
        dlg = BioParamsDialog(self, self.tbl_cbe, self.tbl_rbe,
                               current_name=getattr(self, "active_bio_preset_name", "") or DEFAULT_BIO_PRESET_NAME)
        if dlg.exec():
            # Copiar valores de vuelta a las tablas principales
            for j in range(self.tbl_cbe.columnCount()):
                item = dlg.tbl_cbe.item(0, j)
                if item:
                    self.tbl_cbe.setItem(0, j, QtWidgets.QTableWidgetItem(item.text()))

            for j in range(self.tbl_rbe.columnCount()):
                item = dlg.tbl_rbe.item(0, j)
                if item:
                    self.tbl_rbe.setItem(0, j, QtWidgets.QTableWidgetItem(item.text()))

            self.active_bio_preset_name = dlg.selected_preset_name()
            CBE_list, RBE_list = self._read_cbe_rbe_tables()
            self.CBE, self.RBE = CBE_list, RBE_list
            self.read_ui_into_state()
            self._save_config()
    
    def open_boro_dialog(self):
        current_name = getattr(self, "active_boro_protocol_name", "") or self.combo_proto.currentText()
        dlg = BoroDialog(
            self, self.tbl_boro, current_name=current_name,
            source_mode=getattr(self, "boro_source_mode", "generico"),
            blood_conc=getattr(self, "boro_blood_conc", None),
            blood_conc_err=getattr(self, "boro_blood_conc_err", None),
        )
        if dlg.exec():
            for i in range(2):
                for j in range(len(ORG_ORDER)):
                    item = dlg.tbl.item(i, j)
                    if item:
                        self.tbl_boro.setItem(i, j, QtWidgets.QTableWidgetItem(item.text()))
            selected = dlg.selected_protocol_name()
            if selected and selected != "(manual)":
                self.active_boro_protocol_name = selected
            else:
                self.active_boro_protocol_name = "(manual)"
            # Origen de la concentración: genérico, o a partir de sangre (y en
            # ese caso, con qué concentración/error se calculó cada órgano).
            self.boro_source_mode = dlg.selected_source_mode()
            blood_conc, blood_err = dlg.blood_concentration()
            self.boro_blood_conc = blood_conc
            self.boro_blood_conc_err = blood_err
            self._refresh_proto_combo(selected if selected and selected != "(manual)" else None)
            if selected and selected != "(manual)" and hasattr(self, "combo_proto"):
                self.combo_proto.setCurrentText(selected)
            self._update_active_protocol_label()
            self.read_ui_into_state()
            self._save_config()
            origen_txt = ("a partir de sangre" if self.boro_source_mode == "sangre" else "genérico")
            self.status.setText(
                f"Protocolo de boro activo: "
                f"{self.active_boro_protocol_name if self.active_boro_protocol_name != '(manual)' else 'manual.'} "
                f"({origen_txt})"
            )
    

    # ---------- SPND: flujo desde corrientes ----------
    def open_spnd_config(self):
        dlg = SPNDConfigDialog(self)
        dlg.exec()

    def open_spnd_from_currents(self):
        dlg = SPNDFromCurrentsDialog(self, snapshot=self.last_spnd_currents_snapshot)
        if dlg.exec():
            phi, sigphi = dlg.get_result()
            if phi is None:
                return
            # Guardar snapshot completo para reapertura del diálogo y para el PDF
            self.last_spnd_currents_snapshot = dlg.get_snapshot()
            # RatMaster espera flujo en n/cm²·s y el campo "Error SPND" es ABSOLUTO
            self.input_spnd.setText(f"{phi:.6g}")
            self.input_spnd_err.setText(f"{sigphi:.6g}")

    def open_isoe_organ_dialog(self):
        """
        Abre IsoEOrganParamsDialog para asignar un preset por órgano.
        Auto-detecta el tejido de cada órgano y sugiere el preset compatible.
        El resultado se guarda en self.isoe_organ_assignment.
        """
        # Usar los órganos actualmente cargados en vectores; si no hay, usar ORG_ORDER
        organ_list = list(self.vectors.keys()) if self.vectors else list(ORG_ORDER)

        dlg = IsoEOrganParamsDialog(
            self,
            organ_list=organ_list,
            current_assignment=self.isoe_organ_assignment or {},
        )
        if dlg.exec():
            self.isoe_organ_assignment = dlg.get_assignment()
            # Actualizar etiqueta de estado
            assigned = {o: p for o, p in self.isoe_organ_assignment.items() if p}
            skipped  = [o for o, p in self.isoe_organ_assignment.items() if not p]
            if assigned:
                lines = [f"• {o}: {p}" for o, p in assigned.items()]
                if skipped:
                    lines.append(f"• Sin preset: {', '.join(skipped)} (no se calculará IsoE)")
                self.lbl_isoe_status.setText("\n".join(lines))
                self.lbl_isoe_status.setStyleSheet("color:#1B5E20; font-size:9pt;")
            else:
                self.lbl_isoe_status.setText(
                    "⚠ Ningún órgano tiene preset asignado — no se calculará IsoE."
                )
                self.lbl_isoe_status.setStyleSheet("color:#B71C1C; font-size:9pt;")
            self._save_config()

    def open_isoe_dialog(self):
        """Abre el popup de parámetros MLQ (IsoE) en modo manual (legacy)."""
        current = getattr(self, "isoe_params_manual", {})
        dlg = IsoEDialog(self, current)
        if dlg.exec():
            self.isoe_params_manual = dlg.get_params()

    def _refresh_isoe_preset_combo(self, selected_name=None):
        """Compatibilidad: ya no hay combo de preset global; no hace nada."""
        pass

    def open_isoe_presets_dialog(self):
        """Abre el gestor de biblioteca de presets IsoE (menú ≡)."""
        dlg = IsoEPresetsDialog(self, current_preset_name="Manual")
        if dlg.exec():
            self._save_config()
            self.status.setText("Biblioteca de presets IsoE actualizada.")


    def open_constraints_dialog(self):
        """Abre popup para editar constraints."""
        preset_name = getattr(self, "constraints_preset_name", "") or ""
        blocker = QtCore.QSignalBlocker(self.tbl_cons)
        dlg = ConstraintsDialog(self, self.tbl_cons, preset_name=preset_name)

        if dlg.exec():
            dlg.export_to(self.tbl_cons)
            self.read_ui_into_state()

            cons = self._read_constraints_from_table()
            has_any = bool(np.any(np.asarray(cons, dtype=float) > 0))

            if has_any:
                self.constraints_preset_name = dlg.selected_preset_name()
                if hasattr(self, "lbl_cons_preset"):
                    if self.constraints_preset_name:
                        self.lbl_cons_preset.setText(f"Restricción: {self.constraints_preset_name}")
                    else:
                        self.lbl_cons_preset.setText("Restricción: manual")
                self.status.setText(f"Restricción activa: {self.constraints_preset_name or 'manual'}")
                # Popup de confirmación
                preset_display = self.constraints_preset_name if self.constraints_preset_name else "manual"
                QtWidgets.QMessageBox.information(
                    self,
                    "✔ Restricciones activas",
                    f"Las restricciones de dosis han sido aplicadas correctamente.\n\n"
                    f"Preset activo: {preset_display}\n\n"
                    f"Están listas para ser usadas en el próximo cálculo."
                )
            else:
                self.constraints_preset_name = ""
                if hasattr(self, "lbl_cons_preset"):
                    self.lbl_cons_preset.setText("Restricción: no se seleccionó")
                self.status.setText("Restricción activa: no seleccionada")

            self._save_config()

        del blocker



    # ---------- carga defaults y vectores ----------
    def load_defaults(self):
        proto_name = self.combo_proto.currentText() if hasattr(self, "combo_proto") else "BPA46.5"
        self.active_bio_preset_name = DEFAULT_BIO_PRESET_NAME
        lib = _defaults_from_libs(proto_name, bio_name=self.active_bio_preset_name)
        self.B_arr = lib["B_arr"]; self.B_err = lib["B_err"]
        self.CBE   = lib["CBE"];   self.RBE   = lib["RBE"]

        self.active_boro_protocol_name = proto_name
        self.boro_source_mode = "generico"
        self.boro_blood_conc = None
        self.boro_blood_conc_err = None
        self.populate_boro_rbe_cbe_tables()
        self.populate_constraints_table()
        self.reload_vectors()
        self.status.setText(f"Defaults aplicados (Protocolo={proto_name}) y vectores cargados.")

        self.constraints_preset_name = ""
        if hasattr(self, "lbl_cons_preset"):
            self.lbl_cons_preset.setText("Restricción: no se seleccionó")
        self._update_active_protocol_label()
        self.read_ui_into_state()

    def apply_selected_protocol(self, *_args):
        """Aplica automáticamente el protocolo seleccionado."""
        name = self.combo_proto.currentText() if hasattr(self, "combo_proto") else "BPA46.5"
        if not name:
            return
        try:
            self.read_ui_into_state()
            lib = _defaults_from_libs(name, bio_name=getattr(self, "active_bio_preset_name", DEFAULT_BIO_PRESET_NAME))
            self.B_arr = lib["B_arr"]; self.B_err = lib["B_err"]
            self.CBE   = lib["CBE"];   self.RBE   = lib["RBE"]
            self.active_boro_protocol_name = name
            # Elegir un protocolo de boro directamente desde el combo carga
            # siempre sus valores genéricos de boro; el preset de CBE/RBE
            # activo NO se toca (son independientes — se cambia desde
            # "Parámetros biológicos (CBE / RBE)…", ver BioParamsDialog).
            self.boro_source_mode = "generico"
            self.boro_blood_conc = None
            self.boro_blood_conc_err = None
            self.populate_boro_rbe_cbe_tables()
            self.read_ui_into_state()
            self._update_active_protocol_label()
            self.status.setText(f"Protocolo de boro activo: {name}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"No se pudo aplicar el protocolo: {e}")

    def populate_boro_rbe_cbe_tables(self):
        # Boro
        for j, name in enumerate(ORG_ORDER):
            b  = float(self.B_arr[j]) if (self.B_arr is not None and j < len(self.B_arr)) else 0.0
            be = float(self.B_err[j]) if (self.B_err is not None and j < len(self.B_err)) else 0.0
            self.tbl_boro.setItem(0, j, QtWidgets.QTableWidgetItem(f"{b:.6g}"))
            self.tbl_boro.setItem(1, j, QtWidgets.QTableWidgetItem(f"{be:.6g}"))

        # CBE
        for j in range(len(ORG_ORDER)):
            v = float(self.CBE[j]) if (self.CBE is not None and j < len(self.CBE)) else 1.0
            self.tbl_cbe.setItem(0, j, QtWidgets.QTableWidgetItem(f"{v:.6g}"))

        # RBE
        for j in range(len(ORG_ORDER)):
            v = float(self.RBE[j]) if (self.RBE is not None and j < len(self.RBE)) else 1.0
            self.tbl_rbe.setItem(0, j, QtWidgets.QTableWidgetItem(f"{v:.6g}"))

    def populate_constraints_table(self):
        num_rows = self.tbl_cons.rowCount()
        for i in range(num_rows):
            for j in range(len(ORG_ORDER)):
                val = 0.0
                if (
                    self.Constraints is not None and
                    self.Constraints.shape[0] > i and
                    j < self.Constraints.shape[1]
                ):
                    try:
                        val = float(self.Constraints[i, j])
                    except Exception:
                        val = 0.0

                self.tbl_cons.setItem(i, j, QtWidgets.QTableWidgetItem(f"{val:.6g}"))

    def populate_tables(self):
        self.populate_boro_rbe_cbe_tables()
        self.populate_constraints_table()

    def _safe_float_from_text(self, text, default=0.0):
        try:
            txt = str(text).strip().replace(",", ".")
            if txt == "":
                return float(default)
            return float(txt)
        except Exception:
            return float(default)

    def _table_value(self, table, row, col, default=0.0):
        it = table.item(row, col)
        return self._safe_float_from_text(it.text() if it else "", default)

    def _read_boro_table(self):
        B_list = []
        Berr_list = []
        for j in range(len(ORG_ORDER)):
            B_list.append(self._table_value(self.tbl_boro, 0, j, 0.0))
            Berr_list.append(self._table_value(self.tbl_boro, 1, j, 0.0))
        return np.array(B_list, dtype=float), np.array(Berr_list, dtype=float)

    def _read_cbe_rbe_tables(self):
        CBE_list = []
        RBE_list = []
        for j in range(len(ORG_ORDER)):
            CBE_list.append(self._table_value(self.tbl_cbe, 0, j, 1.0))
            RBE_list.append(self._table_value(self.tbl_rbe, 0, j, 1.0))
        return np.array(CBE_list, dtype=float), np.array(RBE_list, dtype=float)

    def _read_constraints_from_table(self):
        cons = np.zeros((5, len(ORG_ORDER)), dtype=float)
        for i in range(5):
            for j in range(len(ORG_ORDER)):
                cons[i, j] = self._table_value(self.tbl_cons, i, j, 0.0)
        return cons

    def _read_isoe_params_from_ui(self):
        """
        Construye params_by_organ desde self.isoe_organ_assignment.
        Si no hay asignación manual, auto-asigna por tipo de tejido.
        Retorna (params_by_organ, t0_map_by_organ).
        """
        organ_list = list(self.vectors.keys()) if self.vectors else list(ORG_ORDER)

        assignment = self.isoe_organ_assignment or {}
        # Si no hay asignación previa o los órganos cambiaron, auto-asignar
        if not assignment or not any(o in assignment for o in organ_list):
            assignment = auto_assign_presets(organ_list)
            self.isoe_organ_assignment = assignment

        params_by_organ = build_params_by_organ(assignment, USER_ISOE_PRESETS)
        t0_map_by_organ = {}
        from ratmaster.physics.isoe import get_t0_map_for_preset
        for organ, preset_name in assignment.items():
            if preset_name:
                t0_map_by_organ[organ] = get_t0_map_for_preset(preset_name, USER_ISOE_PRESETS)

        return params_by_organ, t0_map_by_organ

    def read_ui_into_state(self):
        B_list, Berr_list = self._read_boro_table()
        CBE_list, RBE_list = self._read_cbe_rbe_tables()
        cons = self._read_constraints_from_table()
        params_by_organ, t0_map_by_organ = self._read_isoe_params_from_ui()
        self.state = {
            "mode_constraints": bool(self.rb_cons.isChecked()),
            "dose_type_index": int(self.combo_type.currentIndex()),
            "spnd": self._safe_float_from_text(self.input_spnd.text(), 0.0),
            "spnd_err_input": self._safe_float_from_text(self.input_spnd_err.text(), 0.0),
            # [FIX] Antes: self._safe_float_from_text(self.input_time.text(), 0.0)
            # solo entendía un número simple. Ahora respeta el formato elegido
            # en self.cmb_time_input_fmt ("s" o "ms" = minutos+segundos).
            "time": (parse_time_input(self.input_time.text(), self._time_input_fmt) or 0.0),
            "time_err": self._safe_float_from_text(self.input_time_err.text(), 0.0),
            "sys_err": self._safe_float_from_text(self.input_sys.text(), 0.0),
            "B": B_list,
            "B_err": Berr_list,
            "CBE": CBE_list,
            "RBE": RBE_list,
            "constraints": cons,
            "constraints_preset_name": getattr(self, "constraints_preset_name", ""),
            "isoe_params_by_organ": params_by_organ,   # {organ: IsoEParams}
            "isoe_t0_map_by_organ": t0_map_by_organ,   # {organ: t0_map}
            "isoe_organ_assignment": dict(self.isoe_organ_assignment or {}),
            "boro_protocol_name": getattr(self, "active_boro_protocol_name", "") or (self.combo_proto.currentText() if hasattr(self, "combo_proto") else ""),
            # Origen de la concentración de boro usada ("generico" o "sangre")
            # y, si vino de sangre, con qué concentración/error se calculó
            # cada órgano — se reporta tal cual en el reporte de resultados.
            "boro_source_mode": getattr(self, "boro_source_mode", "generico"),
            "boro_blood_conc": getattr(self, "boro_blood_conc", None),
            "boro_blood_conc_err": getattr(self, "boro_blood_conc_err", None),
        }
        return self.state

    def validate_state(self, st):
        if not getattr(self, "vectors", {}):
            return False, "No hay vectores cargados."

        spnd_val = float(st.get("spnd", 0.0))
        if spnd_val <= 0:
            return False, "El flujo neutrónico SPND debe ser mayor que cero."

        spnd_err_input = float(st.get("spnd_err_input", 0.0))
        if spnd_err_input < 0:
            return False, "El error SPND no puede ser negativo."

        sys_err = float(st.get("sys_err", 0.0))
        if sys_err < 0:
            return False, "El error sistemático no puede ser negativo."

        if not st.get("mode_constraints", False):
            time_val = float(st.get("time", 0.0))
            if time_val <= 0:
                return False, "En modo tiempo fijo, el tiempo debe ser mayor que cero."
            time_err = float(st.get("time_err", 0.0))
            if time_err < 0:
                return False, "El error de tiempo no puede ser negativo."

        cons = np.array(st.get("constraints", np.zeros((5, len(ORG_ORDER)))), dtype=float)
        if st.get("mode_constraints", False):
            has_any = False
            for j in range(cons.shape[1]):
                dmax = cons[0, j]
                dmean = cons[1, j]
                dmin = cons[2, j]
                vx = cons[3, j]
                dosevx = cons[4, j]

                if min(dmax, dmean, dmin, vx, dosevx) < 0:
                    return False, f"No se permiten restricciones negativas ({ORG_ORDER[j]})."
                if vx > 100:
                    return False, f"El porcentaje (%) no puede ser mayor que 100 ({ORG_ORDER[j]})."
                if (vx > 0 and dosevx <= 0) or (dosevx > 0 and vx <= 0):
                    return False, f"Para usar Vx en {ORG_ORDER[j]} tenés que completar ambos campos: % del volumen más caliente (Vx) y dosis en ese % (Dx)."
                if dmax > 0 or dmean > 0 or dmin > 0 or (vx > 0 and dosevx > 0):
                    has_any = True
            if not has_any:
                return False, "En modo restricciones tenés que definir al menos una restricción válida."

            # Verificar que al menos un constraint definido corresponde a un órgano
            # efectivamente cargado en los vectores actuales.
            loaded_vectors = getattr(self, "vectors", {})
            constrained_organs_missing = []
            constrained_organs_found = []
            for j in range(cons.shape[1]):
                dmax = cons[0, j]; dmean = cons[1, j]; dmin = cons[2, j]
                vx = cons[3, j];   dosevx = cons[4, j]
                has_constraint = dmax > 0 or dmean > 0 or dmin > 0 or (vx > 0 and dosevx > 0)
                if not has_constraint:
                    continue
                organ_logic = ORG_ORDER[j]
                if organ_logic == "Pulmon":
                    exists = any(k in loaded_vectors for k in ("PulmonTotal", "PulmonIzq", "PulmonDer"))
                else:
                    exists = organ_logic in loaded_vectors
                if exists:
                    constrained_organs_found.append(organ_logic)
                else:
                    constrained_organs_missing.append(organ_logic)

            if not constrained_organs_found:
                missing_str = ", ".join(constrained_organs_missing)
                return False, (
                    f"Ningún constraint definido corresponde a un órgano cargado en los vectores actuales.\n\n"
                    f"Órganos con constraint pero SIN vectores: {missing_str}\n\n"
                    f"Verificá que el set de vectores contenga al menos uno de los órganos con constraint, "
                    f"o revisá el protocolo seleccionado."
                )

        if int(st.get("dose_type_index", 0)) == 2:
            # Modo IsoE: verificar que al menos un órgano tiene params asignados
            params_by_organ = st.get("isoe_params_by_organ") or {}
            if not params_by_organ:
                return False, (
                    "En modo IsoE tenés que configurar los parámetros radiobiológicos.\n\n"
                    "Hacé clic en «Configurar parámetros por órgano…» para asignar un "
                    "preset a cada órgano según su tipo de tejido."
                )
            # Validar que los IsoEParams son correctos
            for organ, p in params_by_organ.items():
                try:
                    if not isinstance(p, IsoEParams):
                        return False, f"Parámetros IsoE inválidos para el órgano {organ}."
                except Exception as e:
                    return False, f"Error en parámetros IsoE de {organ}: {e}"

        return True, ""

    def _mark_constraints_manual(self, item=None):
        self.constraints_preset_name = ""
        if hasattr(self, "lbl_cons_preset"):
            self.lbl_cons_preset.setText("Restricción: manual")


    def _refresh_proto_combo(self, selected_name=None):
        current = selected_name or getattr(self, "active_boro_protocol_name", "") or "BPA46.5"
        blocker = QtCore.QSignalBlocker(self.combo_proto) if hasattr(self, "combo_proto") else None
        if hasattr(self, "combo_proto"):
            self.combo_proto.clear()
            self.combo_proto.addItems(sorted(PROTO_LIB.keys()))
            idx = self.combo_proto.findText(current)
            if idx >= 0:
                self.combo_proto.setCurrentIndex(idx)
            elif self.combo_proto.count() > 0:
                self.combo_proto.setCurrentIndex(0)
        if blocker is not None:
            del blocker

    def _update_active_protocol_label(self):
        name = getattr(self, "active_boro_protocol_name", "") or self.combo_proto.currentText() if hasattr(self, "combo_proto") else ""
        if hasattr(self, "lbl_proto_preset"):
            if not name:
                self.lbl_proto_preset.setText("Protocolo activo: no seleccionado")
            elif name == "(manual)":
                self.lbl_proto_preset.setText("Protocolo activo: manual")
            else:
                self.lbl_proto_preset.setText(f"Protocolo activo: {name}")
    # ---------- Config / Vector sets ----------
    def _load_config(self):
        try:
            if self.config_path.exists():
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(cfg, dict):
                    if cfg.get("active_vector_set"):
                        self.active_vector_set = _safe_set_name(cfg.get("active_vector_set"))
                    for name, data in (cfg.get("custom_boro_protocols") or {}).items():
                        try:
                            USER_BORO_PROTOCOLS[name] = _sanitize_boro_protocol_dict(name, data)
                        except Exception:
                            pass
                    for name, mat in (cfg.get("custom_constraint_presets") or {}).items():
                        try:
                            USER_CONSTRAINT_PRESETS[name] = _constraints_matrix_from_serializable(mat)
                        except Exception:
                            pass
                    for name, preset in (cfg.get("custom_isoe_presets") or {}).items():
                        try:
                            if isinstance(preset, dict) and preset.get("params"):
                                IsoEParams(**preset["params"])  # validar
                                USER_ISOE_PRESETS[name] = preset
                        except Exception:
                            pass
                    PROTO_LIB.update(USER_BORO_PROTOCOLS)
                    CONSTRAINT_PRESETS.update(USER_CONSTRAINT_PRESETS)
                    ISOE_PARAM_PRESETS.update(USER_ISOE_PRESETS)
        except Exception:
            pass

    def _save_config(self):
        try:
            custom_boro = {k: PROTO_LIB[k] for k in PROTO_LIB.keys() if not _is_builtin_boro_protocol(k)}
            # [FIX] Antes: np.array(v).tolist() -> formato posicional viejo,
            # vulnerable a desalinearse si después se agrega/quita un órgano
            # de ORG_ORDER. Ahora se serializa por nombre de órgano, igual
            # que save_user_constraint_presets() en data/persistence.py.
            custom_cons = {
                k: _constraints_matrix_to_dict(v, ORG_ORDER)
                for k, v in CONSTRAINT_PRESETS.items()
                if not _is_builtin_constraint_preset(k)
            }
            custom_isoe = {k: v for k, v in USER_ISOE_PRESETS.items() if not _is_builtin_isoe_preset(k)}
            custom_bio = {k: BIO_LIB[k] for k in BIO_LIB.keys() if not _is_builtin_bio_preset(k)}
            cfg = {
                "active_vector_set": self.active_vector_set,
                "custom_boro_protocols": custom_boro,
                "custom_constraint_presets": custom_cons,
                "custom_isoe_presets": custom_isoe,
                "custom_bio_presets": custom_bio,
            }
            self.config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _list_vector_sets(self):
        try:
            return sorted([p.name for p in self.vectors_root.iterdir() if p.is_dir()])
        except Exception:
            return []

    def _ensure_default_vector_set(self):
        """Crea Vectores/DEFAULT y migra (copia) .mat de la raíz si existen."""
        default_dir = self.vectors_root / "DEFAULT"
        default_dir.mkdir(parents=True, exist_ok=True)

        # Migración suave: si hay VectorDoseRate*.mat en base_folder, copiarlos a DEFAULT
        try:
            mats = list(self.base_folder.glob("VectorDoseRate*.mat"))
            if mats:
                for f in mats:
                    dst = default_dir / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
                # guardar meta mínima si no existe
                meta_path = default_dir / "_meta.json"
                if not meta_path.exists():
                    meta = {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "vector_set": "DEFAULT",
                        "note": "Migrado automáticamente desde la carpeta raíz de RatMaster (copia).",
                    }
                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _set_active_vector_set(self, name: str, persist: bool = True):
        name = _safe_set_name(name)
        folder = (self.vectors_root / name).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        self.active_vector_set = name
        self.active_vectors_folder = folder
        if hasattr(self, "lbl_vectors_set"):
            self.lbl_vectors_set.setText(f"Vectores: {name}")
            self.lbl_vectors_set.setToolTip(str(folder))
        if persist:
            self._save_config()

    def open_change_vector_set_dialog(self):
        sets = self._list_vector_sets()
        if not sets:
            sets = ["DEFAULT"]
        cur = self.active_vector_set if self.active_vector_set in sets else (sets[0] if sets else "DEFAULT")
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Cambiar set de vectores", "Elegí el set:", sets, sets.index(cur), False)
        if ok and choice:
            self._set_active_vector_set(choice, persist=True)
            self.reload_vectors()
            self.populate_tables()
            self.status.setText(f"Set de vectores activo: {choice}")


    def open_delete_vector_set_dialog(self):
        sets = [s for s in self._list_vector_sets() if s]
        if not sets:
            QtWidgets.QMessageBox.information(self, "Vectores", "No hay sets para borrar.")
            return

        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Borrar set de vectores",
            "Elegí el set a borrar:",
            sets,
            0,
            False
        )
        if not ok or not choice:
            return

        if choice == "DEFAULT":
            QtWidgets.QMessageBox.warning(self, "Vectores", "No se puede borrar el set DEFAULT.")
            return

        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setWindowTitle("Borrar set")
        msg.setText(f"¿Borrar el set '{choice}' y su carpeta completa?")

        btn_si = msg.addButton("Sí", QtWidgets.QMessageBox.YesRole)
        btn_no = msg.addButton("No", QtWidgets.QMessageBox.NoRole)
        msg.setDefaultButton(btn_no)

        msg.exec()

        if msg.clickedButton() != btn_si:
            return

        try:
            shutil.rmtree(self.vectors_root / choice)
            if self.active_vector_set == choice:
                self._set_active_vector_set("DEFAULT", persist=True)
            self.reload_vectors()
            self.status.setText(f"Set de vectores borrado: {choice}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Vectores", f"No se pudo borrar el set:\n{e}")




    def open_active_vectors_folder(self):
        try:
            url = QtCore.QUrl.fromLocalFile(str(self.active_vectors_folder))
            QtGui.QDesktopServices.openUrl(url)
        except Exception:
            pass

    def _try_load_viz_data(self) -> "dict | None":
        """
        Intenta cargar los datos espaciales guardados por VectorGenDialog.

        Lee _viz_data.npz y _viz_organ_names.json desde la carpeta del set activo.
        Si no existen (set viejo, generado antes de esta versión) retorna None
        y el botón "Visualizar Dosis" quedará deshabilitado.

        Estructura retornada:
            seg_shape             — tuple (X,Y,Z)
            voxel_size_mm         — tuple (sx,sy,sz) en mm
            body_mask_indices     — ndarray (N,3) ijk de vóxeles con label>0
            organ_valid_indices   — {organ: ndarray (Nv,3)} ijk válidos
            phys_dose / bio_dose / iso_dose  — dicts vacíos (se rellenan en run_calc)
        """
        viz_path   = self.active_vectors_folder / "_viz_data.npz"
        names_path = self.active_vectors_folder / "_viz_organ_names.json"

        if not viz_path.exists() or not names_path.exists():
            return None

        try:
            data         = np.load(str(viz_path), allow_pickle=False)
            organ_names  = json.loads(names_path.read_text(encoding="utf-8"))

            seg_shape        = tuple(int(x) for x in data["seg_shape"])
            voxel_size_mm    = tuple(float(x) for x in data["voxel_size_mm"])
            body_mask_indices = data["body_mask_indices"]

            organ_valid_indices: dict = {}
            for i, name in enumerate(organ_names):
                arr_key = f"organ_{i}"
                if arr_key in data.files:
                    organ_valid_indices[name] = data[arr_key]

            # Caso especial pulmón: si hay Izq + Der, armar PulmonTotal
            # para que coincida con la clave que genera compute_bnct
            izq = organ_valid_indices.get("PulmonIzq")
            der = organ_valid_indices.get("PulmonDer")
            if izq is not None and der is not None:
                organ_valid_indices["PulmonTotal"] = np.concatenate(
                    [izq, der], axis=0
                )
            elif izq is not None:
                organ_valid_indices["PulmonTotal"] = izq
            elif der is not None:
                organ_valid_indices["PulmonTotal"] = der

            return {
                "seg_shape":           seg_shape,
                "voxel_size_mm":       voxel_size_mm,
                "body_mask_indices":   body_mask_indices,
                "organ_valid_indices": organ_valid_indices,
                "phys_dose":           {},   # se llena en run_calc()
                "bio_dose":            {},
                "iso_dose":            {},
            }

        except Exception as _e:
            print(f"[WARN] No se pudo cargar _viz_data.npz: {_e}")
            return None

    def reload_vectors(self):
        self.vectors = {}
        # pulmones
        for p in ("PulmonIzq","PulmonDer"):
            try:
                vec = load_vectordose(self.active_vectors_folder, p)
                if vec is not None:
                    self.vectors[p] = vec
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Carga", f"Error cargando {p}: {e}")
        # otros órganos lógicos
        for organ in ORG_ORDER:
            if organ == "Pulmon": continue
            try:
                vec = load_vectordose(self.active_vectors_folder, organ)
                if vec is not None:
                    self.vectors[organ] = vec
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Carga", f"Error cargando {organ}: {e}")
        # PulmonTotal
        L = self.vectors.get("PulmonIzq")
        R = self.vectors.get("PulmonDer")
        if L is not None and R is not None:
            self.vectors["PulmonTotal"] = tuple(np.concatenate([L[i], R[i]]) for i in range(4))
            self.vectors["Tumor"] = self.vectors["PulmonTotal"]
        elif L is not None:
            self.vectors["PulmonTotal"] = L
            self.vectors["Tumor"] = L

        elif R is not None:
            self.vectors["PulmonTotal"] = R
            self.vectors["Tumor"] = R


        # lista UI con checkboxes
        # NOTA: la lista se repoblará con las claves del reporte al terminar run_calc().
        # Aquí solo la limpiamos para reflejar el nuevo set de vectores cargado.
        self.org_list.clear()
        for k in sorted(self.vectors.keys()):
            item = QtWidgets.QListWidgetItem(k)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.org_list.addItem(item)

        # Intentar cargar datos espaciales del set activo para el visor 2D.
        # Si el set fue generado con una versión anterior (sin _viz_data.npz),
        # viz_data queda en None y el botón "Visualizar Dosis" estará deshabilitado.
        self.viz_data = self._try_load_viz_data()
        if self.viz_data is not None:
            n_organs = len(self.viz_data.get("organ_valid_indices", {}))
            print(f"[INFO] viz_data cargado: {n_organs} órganos con coordenadas espaciales.")



    # ---------- Generación de vectores (SEG + Meshtal) ----------
    def pick_default_seg(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Elegir segmentación (.seg)", str(self.base_folder), "SEG (*.seg);;Todos (*.*)")
        if fn:
            self.last_seg_path = fn

    def pick_default_mesh(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Elegir meshtal", str(self.base_folder), "Meshtal (*.msh *.meshtal *.txt);;Todos (*.*)")
        if fn:
            self.last_meshtal_path = fn

    def open_vector_gen_dialog(self):
        dlg = VectorGenDialog(
            parent=self,
            base_folder=self.base_folder,
            defaults={
                "seg_path": self.last_seg_path,
                "meshtal_path": self.last_meshtal_path,
                "voxel_size_mm": self.last_voxel_size_mm,
                "origin_mesh_cm": self.last_origin_mesh_cm,
                "factor_spnd": self.last_factor_spnd,
                "tallies": self.last_tallies,
                "tally_overlay": self.last_tally_overlay,
                "fix_geom": True,
                "vector_set": self.active_vector_set,
                "out_folder": str(self.active_vectors_folder),
            }
        )
        if dlg.exec():
            res = dlg.get_result() or {}
            # guardar defaults
            self.last_seg_path = res.get("seg_path", self.last_seg_path)
            self.last_meshtal_path = res.get("meshtal_path", self.last_meshtal_path)
            self.last_voxel_size_mm = tuple(res.get("voxel_size_mm", self.last_voxel_size_mm))
            self.last_origin_mesh_cm = tuple(res.get("origin_mesh_cm", self.last_origin_mesh_cm))
            self.last_factor_spnd = float(res.get("factor_spnd", self.last_factor_spnd))
            self.last_tallies = list(res.get("tallies", self.last_tallies))
            self.last_tally_overlay = int(res.get("tally_overlay", self.last_tally_overlay))

            # Si VectorGenDialog devuelve datos espaciales, guardarlos para el visor 2D.
            # VectorGenDialog debe exponer en get_result():
            #   "organ_dose": dict  — salida de calcular_dosis_por_organo_trilineal()
            #   "segM":       ndarray — segmentación (X,Y,Z) con labels
            #   "report_ref": dict   — reporte mínimo con PhysVoxel/BioVoxel (puede ser {})
            _organ_dose = res.get("organ_dose")
            _segM       = res.get("segM")
            if _organ_dose is not None and _segM is not None:
                try:
                    self.viz_data = build_viz_data(
                        organ_dose=_organ_dose,
                        report=res.get("report_ref", {}),
                        segM=_segM,
                        voxel_size_mm=self.last_voxel_size_mm,
                    )
                except Exception as _e:
                    self.viz_data = None
                    print(f"[WARN] No se pudo construir viz_data desde VectorGenDialog: {_e}")

            # set activo
            vset = res.get("vector_set")
            if vset:
                self._set_active_vector_set(vset, persist=True)

            # activar set y recargar vectores desde Vectores/<set>/
            try:
                self.reload_vectors()
                self.status.setText(f"Vectores generados y cargados: {self.active_vector_set} ({self.active_vectors_folder})")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "RatMaster", f"Vectores generados, pero falló la recarga:\n{e}")

    # ---------- ejecución cálculo ----------
    def run_calc(self):
        # Invalidar el cache DVH ANTES de empezar cualquier cálculo.
        # Esto garantiza que si el cálculo falla a mitad, el cache quede
        # vacío y no se muestren datos del run anterior.
        self._dvh_cache = {}
        self._dvh_cache_report_id = 0
        self.report = None  # también limpiar el reporte anterior

        # limpiar lienzo
        self.canvas.ax.clear()
        self.canvas.fig.tight_layout(pad=1.5); self.canvas.draw()

        st = self.read_ui_into_state()
        ok, msg = self.validate_state(st)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Datos incompletos o inválidos", msg)
            return

        try:
            spnd_val = float(st["spnd"])
            spnd_err_input = float(st["spnd_err_input"])
            # spnd_err_input es la incertidumbre ABSOLUTA del SPND [n/(cm²·s)].
            # La conversión a fracción relativa (σ/φ) se realiza en compute_bnct,
            # que recibe el valor absoluto. No calcular la fracción aquí (UI).
            time_val = float(st["time"])
            time_err = float(st["time_err"])
            sys_err = float(st["sys_err"])
            mode_constraints = bool(st["mode_constraints"])
            idx = int(st["dose_type_index"])
            self.use_bio_mode = (idx == 1)
            self.use_isoe_mode = (idx == 2)
            isoe_synergy = True

            dose_mode_text = dose_type_display(idx)

            B_list = np.array(st["B"], dtype=float)
            Berr_list = np.array(st["B_err"], dtype=float)
            CBE_list = np.array(st["CBE"], dtype=float)
            RBE_list = np.array(st["RBE"], dtype=float)
            cons = np.array(st["constraints"], dtype=float)

            isoe_params = None
            isoe_params_by_organ = {}
            t0_map_used = dict(DEFAULT_T0_MAP)
            if self.use_isoe_mode:
                isoe_params_by_organ = st.get("isoe_params_by_organ") or {}
                # t0_map_used: usar el primero disponible como fallback global
                t0_maps = st.get("isoe_t0_map_by_organ") or {}
                if t0_maps:
                    t0_map_used = next(iter(t0_maps.values()))
                # isoe_params: fallback para solve_time (usa el primer organ con params)
                if isoe_params_by_organ:
                    isoe_params = next(iter(isoe_params_by_organ.values()))

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Parámetros", f"Error leyendo parámetros: {e}")
            return

        # ejecutar cálculo
        try:
            if self.use_isoe_mode and mode_constraints:
                # Calcular tiempo usando constraints en IsoE (MLQ)
                t_isoe, chosen, _isoe_dmin_info = solve_time_isoe_from_constraints(
                    vectors=self.vectors,
                    organ_order=ORG_ORDER,
                    B_arr=np.array(B_list),
                    spnd_value=spnd_val,
                    sys_error=sys_err,
                    spnd_error=spnd_err_input,   # absoluto [n/(cm²·s)]; compute_bnct calcula eps_S internamente
                    params=isoe_params,
                    synergy=isoe_synergy,
                    constraints_matrix=cons,
                    t0_map=t0_map_used,
                    weight_scheme="sublethal",
                    params_by_organ=isoe_params_by_organ or None,
                )
                report, dsum_phys, dsum_bio, deq = compute_bnct(
                    vectors=self.vectors,
                    organ_order=ORG_ORDER,
                    B_arr=np.array(B_list),
                    B_err_arr=np.array(Berr_list),
                    CBE_arr=np.array(CBE_list),
                    RBE_arr=np.array(RBE_list),
                    spnd_value=spnd_val,
                    time_s=float(t_isoe),
                    time_err=time_err,
                    mode_constraints=False,
                    constraints_matrix=None,
                    sys_error=sys_err,
                    spnd_error=spnd_err_input,   # absoluto [n/(cm²·s)]
                    dose_for_limits=("bio" if self.use_bio_mode else "phys"),
                    # El tiempo proviene de constraints IsoE → spnd cancela en dosis
                    time_from_constraints=True,
                )
                if "meta" in report:
                    report["meta"]["mode"] = "constraints"
                    report["meta"]["chosen_constraint"] = chosen
                    report["meta"]["constraints_used"] = summarize_constraints_matrix(cons, ORG_ORDER)
                    if _isoe_dmin_info is not None:
                        report["meta"]["dmin_conflict"] = _isoe_dmin_info
            else:
                report, dsum_phys, dsum_bio, deq = compute_bnct(
                    vectors=self.vectors,
                    organ_order=ORG_ORDER,
                    B_arr=np.array(B_list),
                    B_err_arr=np.array(Berr_list),
                    CBE_arr=np.array(CBE_list),
                    RBE_arr=np.array(RBE_list),
                    spnd_value=spnd_val,
                    time_s=time_val,
                    time_err=time_err,
                    mode_constraints=mode_constraints,
                    constraints_matrix=cons if mode_constraints else None,
                    sys_error=sys_err,
                    spnd_error=spnd_err_input,   # absoluto [n/(cm²·s)]
                    dose_for_limits=("bio" if self.use_bio_mode else "phys"),
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error de cálculo", str(e))
            return

        # Completar metadatos para resultados / reporte
        if "meta" not in report:
            report["meta"] = {}
        report["meta"]["spnd_abs_err"] = float(spnd_err_input)
        # time_err ya fue calculado correctamente dentro de compute_bnct:
        #   modo constraints → time_used × eps_S   (SPND domina σ(t))
        #   modo tiempo fijo → incertidumbre absoluta del usuario
        # No sobreescribir aquí (evita inconsistencia entre lo que ven las
        # varianzas de dosis y lo que muestra el reporte).
        report["meta"]["boro_protocol_name"] = _resolve_boro_protocol_name(st.get("boro_protocol_name") or getattr(self, "active_boro_protocol_name", "") or (self.combo_proto.currentText() if hasattr(self, "combo_proto") else "Manual"))
        # Origen de la concentración de boro: "generico" (valores del
        # protocolo) o "sangre" (calculada a partir de una medición en sangre
        # y la relación Tejido/Sangre del protocolo). Si es "sangre", se
        # reporta también la concentración y el error usados.
        report["meta"]["boro_source_mode"] = st.get("boro_source_mode", getattr(self, "boro_source_mode", "generico"))
        report["meta"]["boro_blood_conc"] = st.get("boro_blood_conc", getattr(self, "boro_blood_conc", None))
        report["meta"]["boro_blood_conc_err"] = st.get("boro_blood_conc_err", getattr(self, "boro_blood_conc_err", None))
        report["meta"]["dose_mode_text"] = dose_mode_text
        if mode_constraints:
            report["meta"]["constraints_used"] = summarize_constraints_matrix(cons, ORG_ORDER)
        report["meta"]["isoe_organ_assignment"] = {
            o: p for o, p in (self.isoe_organ_assignment or {}).items()
        }
        # Adjuntar datos de corrientes si el flujo vino del asistente SPND
        if self.last_spnd_currents_snapshot:
            report["meta"]["spnd_from_currents"] = self.last_spnd_currents_snapshot
        report["meta"]["isoe_t0_map"] = dict(t0_map_used)

        # seleccionar métricas/visualización según modo
        if self.use_isoe_mode:
            # eps_rel_for_isoe fue calculado en compute_bnct con la lógica correcta:
            #   modo constraints → solo eps_sys  (eps_S y eps_time cancelan en D[v])
            #   modo tiempo fijo → sqrt(eps_S² + eps_time² + eps_sys²)
            # Se lee del reporte en vez de recalcularlo aquí (física en physics, no en UI).
            eps_rel_isoe = float(
                report.get("meta", {}).get("eps_rel_for_isoe", float(sys_err))
            )

            # Calcular IsoE (MLQ) — cada órgano usa sus propios parámetros.
            # params=None garantiza que órganos sin preset asignado se OMITEN.
            isoe_report, Dsum_isoe = compute_isoe_from_report(
                report,
                params=None,
                synergy=isoe_synergy,
                t0_map=t0_map_used,
                weight_scheme="sublethal",
                params_by_organ=isoe_params_by_organ or None,
                eps_rel=eps_rel_isoe,
            )
            # Incrustar voxeles IsoE en el mismo reporte para reutilizar UI
            report["IsoVoxel"] = isoe_report.get("IsoVoxel", {})
            report["IsoE_meta"] = isoe_report.get("meta", {})
            dsum_selected = Dsum_isoe
            # si venimos de constraints, guardar "achieved_value" para el constraint elegido
            try:
                if mode_constraints and "meta" in report and report["meta"].get("chosen_constraint"):
                    ch = report["meta"]["chosen_constraint"]
                    org = ch.get("org") or ch.get("organ")
                    typ = ch.get("type") or ch.get("metric")
                    ach = achieved_value_for_constraint(report.get("IsoVoxel", {}), org, typ)
                    if ach is not None:
                        report["meta"]["chosen_constraint"]["achieved_value"] = float(ach)
            except Exception:
                pass
        else:
            # seleccionar métricas según modo
            dsum_selected = dsum_bio if self.use_bio_mode else dsum_phys

        # completar achieved_value del constraint elegido con el mapa voxel final
        try:
            if mode_constraints and "meta" in report and report["meta"].get("chosen_constraint"):
                ch = report["meta"]["chosen_constraint"]
                org = ch.get("org") or ch.get("organ")
                typ = ch.get("type") or ch.get("metric")
                final_voxmap = report.get("IsoVoxel" if self.use_isoe_mode else ("BioVoxel" if self.use_bio_mode else "PhysVoxel"), {})
                ach = achieved_value_for_constraint(final_voxmap, org, typ)
                if ach is not None:
                    report["meta"]["chosen_constraint"]["achieved_value"] = float(ach)
        except Exception:
            pass

        # --- Popup de resultado (modo constraints) ---
        if mode_constraints:
            meta = report.get("meta", {})
            ch = meta.get("chosen_constraint")

            if ch:
                org = ch.get("org", "Desconocido")
                t   = ch.get("time_computed", meta.get("time", 0.0))
                typ = ch.get("type", "Constraint")
                lim = ch.get("limit_value", 0.0)
                ach = float(ch.get("achieved_value", 0.0) or 0.0)

                t_sigma = float(report.get("meta", {}).get("time_err", 0.0))

                msg = QtWidgets.QMessageBox(self)
                msg.setWindowTitle("Tiempo óptimo (Restricciones de dosis)")
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText(
                    f"<h1>Tiempo usado para el cálculo: {format_time(t, t_sigma, 's')}</h1>"
                    f"<p><b>Órgano / restricción limitante:</b> {org}</p>"
                    f"<p><b>Tipo:</b> {typ} &nbsp;&nbsp; <b>Límite:</b> {lim} {dose_axis_unit(self.use_bio_mode, self.use_isoe_mode)}</p>"
                    f"<p><b>Valor logrado:</b> {ach:.3f} {dose_axis_unit(self.use_bio_mode, self.use_isoe_mode)}</p>"
                )
                msg.exec()

        # guardar reporte y construir cache DVH
        # El cache_report_id es el id() del nuevo objeto report,
        # así _plot_dvh_for_keys puede verificar que trabaja con el
        # reporte correcto y no con uno anterior.
        self.report, self.dsum, self.deq = report, dsum_selected, deq

        # Construir el cache DVH ANTES de abrir el diálogo:
        # mientras el usuario lee la tabla de resultados, los sorts ya
        # están hechos. Cuando cierre el diálogo el plot es instantáneo.
        self._dvh_cache_report_id = id(report)
        self._build_dvh_cache(report)

        # Inyectar las dosis del reporte fresco en viz_data para el visor 2D.
        # viz_data ya tiene las coordenadas espaciales (cargadas en reload_vectors);
        # aquí se añaden los arrays de dosis que compute_bnct acaba de calcular.
        # Las claves de PhysVoxel deben coincidir con organ_valid_indices para
        # que la superposición funcione (se usan los mismos nombres normalizados).
        if self.viz_data is not None:
            self.viz_data["phys_dose"] = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in report.get("PhysVoxel", {}).items()
            }
            self.viz_data["bio_dose"] = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in report.get("BioVoxel", {}).items()
            }
            self.viz_data["iso_dose"] = {
                k: np.asarray(v, dtype=np.float64)
                for k, v in report.get("IsoVoxel", {}).items()
            }

        dlg = ResultsDialog(
            self, report, dsum_selected, deq, dose_mode_text,
            use_bio=self.use_bio_mode,
            is_isoe=self.use_isoe_mode,
            viz_data=self.viz_data,
            isoe_params_by_organ=isoe_params_by_organ if self.use_isoe_mode else {},
        )
        dlg.exec()

        # status extra para IsoE + constraints
        if self.use_isoe_mode and mode_constraints and "meta" in report and report["meta"].get("chosen_constraint"):
            ch = report["meta"]["chosen_constraint"]
            self.status.setText(
                f"IsoE constraints — t={format_time(report.get('meta',{}).get('time',0), report.get('meta',{}).get('time_err',0), 's')} — "
                f"{ch.get('org')} / {ch.get('type')} ≤ {ch.get('limit_value')} {dose_axis_unit(self.use_bio_mode, self.use_isoe_mode)} "
                f"(preset IsoE: {report.get('meta',{}).get('isoe_preset_name','Manual')})"
            )

        # ── DVH en ventana principal ──────────────────────────────────────────
        # Las claves del voxmap son las "verdaderas" del reporte (pueden diferir
        # de vectors.keys() en el caso del pulmón: "PulmonTotal" vs "Pulmon").
        # Repoblamos org_list con estas claves para que checkbox ↔ cache sean 1:1.
        voxmap_keys = list(report.get(
            "IsoVoxel" if self.use_isoe_mode else ("BioVoxel" if self.use_bio_mode else "PhysVoxel"),
            {}
        ).keys())

        # Mapa de colores fijo (índice = posición en voxmap_keys)
        from ratmaster.constants import ORGAN_COLORS as _OC
        self._organ_color_map = {
            key: _OC[idx % len(_OC)]
            for idx, key in enumerate(voxmap_keys)
        }

        # Repoblar org_list con exactamente las claves del voxmap
        # (así el texto del checkbox siempre coincide con una clave del cache)
        self.org_list.blockSignals(True)
        self.org_list.clear()
        for key in voxmap_keys:
            it = QtWidgets.QListWidgetItem(key)
            it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)   # sin check = mostrar todos
            self.org_list.addItem(it)
        self.org_list.blockSignals(False)

        self._update_org_list_colors()
        # Cache ya listo (construido antes del diálogo) → plot inmediato
        self._plot_dvh_for_keys(voxmap_keys, full_render=True)

        # status
        meta = report.get("meta", {})
        if meta.get("mode") == "constraints" and meta.get("chosen_constraint"):
            ch = meta["chosen_constraint"]
            ach = float(ch.get('achieved_value', 0.0) or 0.0)
            self.status.setText(f"OK — t={meta.get('time',0):.2f}s | Limitante: {ch.get('org','?')} {ch.get('type','?')} (límite {ch.get('limit_value','?')}, logrado {ach:.3f})")
        else:
            self.status.setText(f"OK — t={meta.get('time',0):.2f}s (modo tiempo fijo)")

    # ---------- DVH — método unificado (exact voxel-by-voxel) ----------
    def _build_dvh_cache(self, report: dict) -> None:
        """
        Pre-calcula y almacena los arrays (s, vol) de todos los órganos del reporte.

        Se llama UNA SOLA VEZ al recibir resultados nuevos.
        El cache queda ligado al id() del reporte activo: si el reporte cambia
        (nuevo cálculo), run_calc() ya limpió el cache antes de llegar acá,
        así que no hay posibilidad de mezclar datos de distintos cálculos.
        """
        self._dvh_cache = {}
        voxmap = report.get(
            "IsoVoxel" if self.use_isoe_mode else ("BioVoxel" if self.use_bio_mode else "PhysVoxel"),
            {}
        )
        for key, vox in voxmap.items():
            arr = np.array(vox, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            s, vol = build_dvh(arr)
            s, vol = dvh_extend_to_zero(s, vol)
            if s.size > 0:
                self._dvh_cache[key] = (s, vol)

    def _plot_dvh_for_keys(self, keys, full_render: bool = False):
        """
        Grafica el DVH para las claves dadas usando el cache pre-calculado.

        Args:
            keys:        lista de claves de órganos a graficar.
            full_render: si True, aplica tight_layout y draw() completo.
                         Si False (default para updates de checkbox), usa
                         draw_idle() que es ~10x más rápido para cambios
                         interactivos porque no recalcula el layout.
        """
        if not self.report:
            return
        # Verificar que el cache corresponde al reporte activo
        if id(self.report) != self._dvh_cache_report_id:
            # Esto no debería pasar nunca, pero si ocurre re-construimos el cache
            self._build_dvh_cache(self.report)
            self._dvh_cache_report_id = id(self.report)

        color_map = getattr(self, "_organ_color_map", {})
        _FALLBACK_COLORS = [
            "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
            "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
        ]
        self.canvas.ax.clear()
        plotted = []
        fallback_idx = 0
        for key in keys:
            if key not in self._dvh_cache:
                continue                                    # órgano sin datos, saltar
            s, vol = self._dvh_cache[key]                  # O(1): ya calculado
            color = color_map.get(key, _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)])
            if key not in color_map:
                fallback_idx += 1
            self.canvas.ax.plot(s, vol, label=key, color=color)
            plotted.append(key)
        xlab = f"Dosis ({dose_axis_unit(self.use_bio_mode, self.use_isoe_mode)})"
        title_base = (
            "DVH — Dosis Isoefectiva (Photon IsoE)" if self.use_isoe_mode
            else ("DVH — Dosis Equivalente pesada por RBE" if self.use_bio_mode else "DVH — Dosis Física")
        )
        self.canvas.ax.set_xlabel(xlab)
        self.canvas.ax.set_ylabel("Volumen (%)")
        self.canvas.ax.set_title(title_base)
        self.canvas.ax.set_xlim(left=0)
        self.canvas.ax.set_ylim(0, 105)
        self.canvas.ax.grid(True, which='major', color='#E0E0E0', linewidth=0.8, alpha=0.9)
        self.canvas.ax.grid(True, which='minor', color='#EEEEEE', linewidth=0.4, alpha=0.7)
        self.canvas.ax.minorticks_on()
        self.canvas.ax.set_axisbelow(True)
        if plotted:
            self.canvas.ax.legend(loc="best", fontsize=8, framealpha=0.85, edgecolor='#CCCCCC')
        if full_render:
            # Primera vez o tras cambio de modo: recalcular layout completo
            self.canvas.fig.tight_layout(pad=1.5)
            self.canvas.draw()
        else:
            # Update interactivo (checkbox): solo redibujar las líneas,
            # sin recalcular el layout → ~10x más rápido
            self.canvas.draw_idle()

    def _update_org_list_colors(self):
        """Actualiza el color de fondo de cada ítem de la lista usando el mapa de colores fijo."""
        color_map = getattr(self, "_organ_color_map", {})
        _FALLBACK_COLORS = [
            "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
            "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
        ]
        fallback_idx = 0
        for i in range(self.org_list.count()):
            it = self.org_list.item(i)
            if it is None:
                continue
            key = it.text()
            checked = it.checkState() == QtCore.Qt.Checked
            # Obtener color fijo del órgano
            color_hex = color_map.get(key, _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)])
            if key not in color_map:
                fallback_idx += 1
            if checked:
                qc = QtGui.QColor(color_hex)
                r = min(255, 210 + int(qc.red()   * 0.18))
                g = min(255, 210 + int(qc.green() * 0.18))
                b = min(255, 210 + int(qc.blue()  * 0.18))
                it.setBackground(QtGui.QColor(r, g, b))
                it.setForeground(QtGui.QColor(color_hex).darker(170))
                f = it.font(); f.setBold(True); it.setFont(f)
            else:
                it.setBackground(QtGui.QColor("#FFFFFF"))
                it.setForeground(QtGui.QColor("#90A4AE"))
                f = it.font(); f.setBold(False); it.setFont(f)

    def _on_organ_check_changed(self, _item=None):
        """
        Actualiza el DVH cada vez que se marca o desmarca un órgano.
        Lógica: sin ningún check = mostrar todos; con checks = mostrar solo los marcados.
        """
        self._update_org_list_colors()
        keys = []
        for i in range(self.org_list.count()):
            it = self.org_list.item(i)
            if it and it.checkState() == QtCore.Qt.Checked:
                keys.append(it.text())
        if not keys:
            # Sin selección: mostrar todos los disponibles en el reporte actual
            if self.report:
                all_keys = list(self.report.get(
                    "IsoVoxel" if self.use_isoe_mode else ("BioVoxel" if self.use_bio_mode else "PhysVoxel"),
                    {}
                ).keys())
                self._plot_dvh_for_keys(all_keys)  # full_render=False (default): draw_idle
            else:
                self._plot_dvh_for_keys([])
        else:
            self._plot_dvh_for_keys(keys)  # full_render=False (default): draw_idle

    def show_all_dvh(self):
        """Desmarca todos los órganos → sin selección = mostrar todos los DVH."""
        if not self.report:
            return
        self.org_list.blockSignals(True)
        for i in range(self.org_list.count()):
            it = self.org_list.item(i)
            if it:
                it.setCheckState(QtCore.Qt.Unchecked)
        self.org_list.blockSignals(False)
        self._update_org_list_colors()
        # Plotear directamente con full_render=True (reset de vista completo)
        all_keys = list(self._dvh_cache.keys())
        self._plot_dvh_for_keys(all_keys, full_render=True)
# ------------------- main -------------------
