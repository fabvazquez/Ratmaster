"""
ui/dialogs/vector_gen.py
========================
Diálogo generador de vectores de dosis desde archivos SEG + Meshtal.

Permite:
  - Elegir archivo SEG (.seg) y meshtal (.msh/.meshtal) de MCNP/FMESH.
  - Configurar tamaño de vóxel, origen de mesh y factor SPND.
  - Ejecutar el pipeline bnct_union y exportar .mat por órgano.
  - Mostrar overlay SEG+MESH para verificar alineación.
"""

import subprocess
import unicodedata
import re
import numpy as np
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ratmaster.app_paths import app_install_dir, user_app_dir


from ratmaster.data.vector_loader import _safe_set_name
class VectorGenDialog(QtWidgets.QDialog):
    """
    Dialogo para:
    - elegir SEG (.seg) y meshtal (.msh/.meshtal)
    - definir voxel size, origen de mesh, factor SPND
    - correr el pipeline bnct_union -> exportar .mat en carpeta RatMaster
    - mostrar overlay SEG+MESH para verificar alineación
    """
    def __init__(self, parent, base_folder: Path, defaults: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Generar vectores de dosis (SEG + Meshtal MCNP)")
        self.resize(760, 420)
        self.base_folder = Path(base_folder).resolve()
        self.defaults = defaults or {}

        self._result = None  # dict con paths/params + outputs

        lay = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        # --- SEG ---
        self.in_seg = QtWidgets.QLineEdit(self.defaults.get("seg_path", ""))
        btn_seg = QtWidgets.QPushButton("Elegir…")
        btn_seg.clicked.connect(self._pick_seg)
        row_seg = QtWidgets.QHBoxLayout()
        row_seg.addWidget(self.in_seg)
        row_seg.addWidget(btn_seg)
        w_seg = QtWidgets.QWidget()
        w_seg.setLayout(row_seg)
        form.addRow("Segmentación (.seg):", w_seg)

        # --- MESHTAL ---
        self.in_mesh = QtWidgets.QLineEdit(self.defaults.get("meshtal_path", ""))
        btn_mesh = QtWidgets.QPushButton("Elegir…")
        btn_mesh.clicked.connect(self._pick_mesh)
        row_mesh = QtWidgets.QHBoxLayout()
        row_mesh.addWidget(self.in_mesh)
        row_mesh.addWidget(btn_mesh)
        w_mesh = QtWidgets.QWidget()
        w_mesh.setLayout(row_mesh)
        form.addRow("Meshtal MCNP (.msh/.meshtal):", w_mesh)

        # --- voxel size ---
        vs = self.defaults.get("voxel_size_mm", (0.78125, 0.3325, 0.3325))
        self.in_vx = QtWidgets.QLineEdit(str(vs[0]))
        self.in_vy = QtWidgets.QLineEdit(str(vs[1]))
        self.in_vz = QtWidgets.QLineEdit(str(vs[2]))
        row_vs = QtWidgets.QHBoxLayout()
        for w in (self.in_vx, self.in_vy, self.in_vz):
            w.setMaximumWidth(120)
            row_vs.addWidget(w)
        w_vs = QtWidgets.QWidget()
        w_vs.setLayout(row_vs)
        form.addRow("Voxel size [mm] (x, y, z):", w_vs)

        # --- origin ---
        org = self.defaults.get("origin_mesh_cm", (230.0, -2.25, -5.0))
        self.in_ox = QtWidgets.QLineEdit(str(org[0]))
        self.in_oy = QtWidgets.QLineEdit(str(org[1]))
        self.in_oz = QtWidgets.QLineEdit(str(org[2]))
        row_org = QtWidgets.QHBoxLayout()
        for w in (self.in_ox, self.in_oy, self.in_oz):
            w.setMaximumWidth(120)
            row_org.addWidget(w)
        w_org = QtWidgets.QWidget()
        w_org.setLayout(row_org)
        form.addRow("Origin mesh [cm] (x0, y0, z0):", w_org)

        # --- factor spnd ---
        self.in_factor = QtWidgets.QLineEdit(str(self.defaults.get("factor_spnd", 3.6006e-9)))
        self.in_factor.setMaximumWidth(160)
        form.addRow("Factor SPND (ver bnct_union):", self.in_factor)

        # --- tallies ---
        self.in_tallies = QtWidgets.QLineEdit(
            ",".join(str(x) for x in self.defaults.get("tallies", [14, 24, 34, 44]))
        )
        self.in_tallies.setMaximumWidth(220)
        form.addRow("Tallies a usar:", self.in_tallies)

        # --- tally overlay ---
        self.cmb_overlay = QtWidgets.QComboBox()
        self.cmb_overlay.addItems([str(x) for x in self.defaults.get("tallies", [14, 24, 34, 44])])
        self.cmb_overlay.setCurrentText(str(self.defaults.get("tally_overlay", 14)))
        self.cmb_overlay.setMaximumWidth(120)
        form.addRow("Tally para overlay:", self.cmb_overlay)

        # --- opciones geométricas ---
        self.chk_fixgeom = QtWidgets.QCheckBox("Aplicar corrección geométrica (swap YZ + rotate)")
        self.chk_fixgeom.setChecked(bool(self.defaults.get("fix_geom", True)))
        form.addRow("SEG:", self.chk_fixgeom)

        # --- set de vectores (subcarpeta) ---
        self.in_set = QtWidgets.QLineEdit()
        self.in_set.setPlaceholderText("Ej: 2026-02-25_tapita_pix078")
        self.in_set.clear()
        form.addRow("Set de vectores:", self.in_set)

        # --- salida ---
        self.lbl_out = QtWidgets.QLabel("")
        self.lbl_out.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_out.setText("(ingresá un nombre para el nuevo set)")
        form.addRow("Salida (.mat):", self.lbl_out)

        self.in_set.textChanged.connect(self._update_output_label)

        note = QtWidgets.QLabel(
            "Se creará (o usará) una subcarpeta en Vectores/<set>/ con .mat + _meta.json + "
            "overlay_check.png. No pisa otros sets."
        )
        note.setStyleSheet("color: #555;")
        lay.addWidget(note)

        # --- botones ---
        btns = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Generar")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px;")
        btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_align = QtWidgets.QPushButton("🔍  Ver alineación")
        self.btn_align.setToolTip(
            "Abre el visor interactivo de superposición SEG ↔ Mesh para\n"
            "verificar la alineación antes o después de generar vectores.\n"
            "Requiere que los archivos SEG y meshtal estén seleccionados."
        )
        self.btn_align.setEnabled(False)   # se habilita cuando ambos archivos existen
        btns.addStretch(1)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_align)
        btns.addWidget(btn_cancel)
        lay.addLayout(btns)

        btn_cancel.clicked.connect(self.reject)
        self.btn_run.clicked.connect(self._run_pipeline)
        self.btn_align.clicked.connect(self._open_alignment_viewer)

        # Habilitar btn_align en tiempo real cuando ambos campos tienen texto
        self.in_seg.textChanged.connect(self._update_align_btn)
        self.in_mesh.textChanged.connect(self._update_align_btn)

    def _load_bu(self):
        """
        Carga bnct_union.py dinámicamente (mismo mecanismo que _run_pipeline).
        Retorna el módulo, o None si no se encontró (ya muestra el error al usuario).
        """
        import importlib.util

        search_dirs = [
            app_install_dir(),
            app_install_dir().parent,
            user_app_dir(),
        ]
        for d in search_dirs:
            candidate = d / "bnct_union.py"
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("bnct_union", str(candidate))
                bu = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(bu)
                    return bu
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, "bnct_union — Error al cargar",
                        f"Se encontró bnct_union.py en:\n  {candidate}\n\n"
                        f"Pero falló al cargarlo:\n{e}"
                    )
                    return None

        paths_tried = "\n".join(f"  • {d / 'bnct_union.py'}" for d in search_dirs)
        QtWidgets.QMessageBox.critical(
            self, "bnct_union no encontrado",
            "No se encontró bnct_union.py en ninguna de estas ubicaciones:\n\n"
            f"{paths_tried}\n\n"
            "Colocá bnct_union.py en la misma carpeta donde está la carpeta "
            "ratmaster/ (o junto al ejecutable RatMaster.exe)."
        )
        return None

    def _update_align_btn(self):
        """Habilita 'Ver alineación' cuando ambos archivos están seleccionados."""
        seg_ok  = bool(self.in_seg.text().strip())
        mesh_ok = bool(self.in_mesh.text().strip())
        self.btn_align.setEnabled(seg_ok and mesh_ok)

    def _open_alignment_viewer(self):
        """
        Carga el SEG y el meshtal desde los campos actuales de la UI y abre
        MeshSegViewerDialog para verificar la alineación interactivamente.
        Funciona tanto ANTES como DESPUÉS de generar vectores.
        """
        from ratmaster.ui.dialogs.results import MeshSegViewerDialog

        bu = self._load_bu()
        if bu is None:
            return

        seg_path    = self.in_seg.text().strip()
        meshtal_path = self.in_mesh.text().strip()

        if not seg_path or not Path(seg_path).exists():
            QtWidgets.QMessageBox.warning(
                self, "Archivo no encontrado",
                f"No se encontró el archivo SEG:\n{seg_path}"
            )
            return
        if not meshtal_path or not Path(meshtal_path).exists():
            QtWidgets.QMessageBox.warning(
                self, "Archivo no encontrado",
                f"No se encontró el meshtal:\n{meshtal_path}"
            )
            return

        try:
            voxel_size_mm  = self._parse_triplet(
                self.in_vx.text(), self.in_vy.text(), self.in_vz.text()
            )
            origin_mesh_cm = self._parse_triplet(
                self.in_ox.text(), self.in_oy.text(), self.in_oz.text()
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Parámetros inválidos",
                f"Error al leer voxel size u origin:\n{e}"
            )
            return

        # Mostrar cursor de espera mientras carga
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            Seg    = bu.load_seg_with_lut(seg_path, pixel_size_mm=voxel_size_mm)
            segM   = Seg["Matrix"]
            if self.chk_fixgeom.isChecked():
                segM = bu.rotate_segmentation(bu.swap_YZ(segM))
            meshes = bu.read_meshtal_all(meshtal_path)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Error al cargar datos",
                f"No se pudieron cargar los archivos:\n{e}"
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not meshes:
            QtWidgets.QMessageBox.warning(
                self, "Meshtal vacío",
                "No se encontraron meshes en el archivo. "
                "Verificá que el path sea correcto."
            )
            return

        dlg = MeshSegViewerDialog(
            self,
            segM           = segM,
            meshes         = meshes,
            voxel_size_mm  = voxel_size_mm,
            origin_mesh_cm = origin_mesh_cm,
            lut            = Seg["LUT"],
        )
        dlg.exec()

    def _pick_seg(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Elegir segmentación",
            str(self.base_folder),
            "SEG (*.seg);;Todos (*.*)"
        )
        if fn:
            self.in_seg.setText(fn)

    def _pick_mesh(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Elegir meshtal",
            str(self.base_folder),
            "Meshtal (*.msh *.meshtal *.txt);;Todos (*.*)"
        )
        if fn:
            self.in_mesh.setText(fn)

    def _parse_triplet(self, a, b, c, cast=float):
        return (cast(a), cast(b), cast(c))

    def _parse_int_list(self, s: str):
        parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
        out = []
        for p in parts:
            out.append(int(float(p)))
        return out

    def get_result(self):
        return self._result

    def _update_output_label(self):
        set_name = self.in_set.text().strip()

        if not set_name:
            self.lbl_out.setText("(ingresá un nombre para el nuevo set)")
            return

        if set_name.upper() == "DEFAULT":
            self.lbl_out.setText("(no se permite usar DEFAULT)")
            return

        out_folder = (Path(self.base_folder) / "Vectores" / set_name).resolve()
        self.lbl_out.setText(str(out_folder))

    def _create_busy_dialog(self, title: str, text: str):
        dlg = QtWidgets.QProgressDialog(self)
        dlg.setWindowTitle(title)
        dlg.setLabelText(text)
        dlg.setRange(0, 0)  # modo indeterminado
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(460)

        try:
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
            dlg.setWindowIcon(icon)
        except Exception:
            pass

        dlg.show()
        QtWidgets.QApplication.processEvents()
        return dlg

    def _set_busy_message(self, dlg, text: str):
        if dlg is None:
            return
        try:
            dlg.setLabelText(text)
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

    def _run_pipeline(self):
        """
        Importa bnct_union.py dinámicamente (via _load_bu) y ejecuta el
        pipeline completo: SEG → Meshtal → interpolación → exportar .mat.
        """
        bu = self._load_bu()
        if bu is None:
            return

        seg_path = self.in_seg.text().strip()
        meshtal_path = self.in_mesh.text().strip()

        if not seg_path or not Path(seg_path).exists():
            QtWidgets.QMessageBox.warning(self, "Faltan datos", "Elegí un archivo .seg válido.")
            return

        if not meshtal_path or not Path(meshtal_path).exists():
            QtWidgets.QMessageBox.warning(self, "Faltan datos", "Elegí un archivo meshtal válido.")
            return

        try:
            voxel_size_mm = self._parse_triplet(
                self.in_vx.text(), self.in_vy.text(), self.in_vz.text(), float
            )
            origin_mesh_cm = self._parse_triplet(
                self.in_ox.text(), self.in_oy.text(), self.in_oz.text(), float
            )
            factor_spnd = float(self.in_factor.text())
            tallies = self._parse_int_list(self.in_tallies.text())
            tally_overlay = int(self.cmb_overlay.currentText())

            raw_set_name = self.in_set.text().strip()
            if not raw_set_name:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Parámetros",
                    "Ingresá un nombre para el nuevo set de vectores."
                )
                return

            vector_set = _safe_set_name(raw_set_name)
            if not vector_set:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Parámetros",
                    "El nombre del set no es válido."
                )
                return

            if vector_set.upper() == "DEFAULT":
                QtWidgets.QMessageBox.warning(
                    self,
                    "Parámetros",
                    "No se permite usar 'DEFAULT' como nombre del nuevo set."
                )
                return

            out_folder = (Path(self.base_folder) / "Vectores" / vector_set).resolve()
            out_folder.mkdir(parents=True, exist_ok=True)

            # actualizar label salida
            self.lbl_out.setText(str(out_folder))

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Parámetros", f"Error leyendo parámetros:\n{e}")
            return

        busy = None
        prev_enabled = self.btn_run.isEnabled()

        try:
            self.btn_run.setEnabled(False)

            busy = self._create_busy_dialog(
                "Generando vectores de dosis",
                "Preparando archivos…\n\nEsto puede tardar unos momentos."
            )

            self._set_busy_message(
                busy,
                "Preparando generación de vectores…"
            )

            # --- 1) SEG ---
            self._set_busy_message(
                busy,
                "Leyendo la segmentación (.seg)…\n\nSe están generando los nuevos vectores de dosis. Esperá por favor."
            )
            try:
                Seg = bu.load_seg_with_lut(seg_path, pixel_size_mm=voxel_size_mm)
                segM = Seg["Matrix"]
                lut = Seg["LUT"]
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "SEG", f"No se pudo leer la segmentación:\n{e}")
                return

            # corrección geométrica opcional (igual a tu demo)
            self._set_busy_message(
                busy,
                "Aplicando corrección geométrica de la segmentación…\n\nSwap/rotate en progreso."
            )
            try:
                if self.chk_fixgeom.isChecked():
                    segM = bu.swap_YZ(segM)
                    sx, sy, sz = Seg["VoxelSize_mm"]
                    Seg["VoxelSize_mm"] = (sx, sz, sy)
                    segM = bu.rotate_segmentation(segM)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "SEG",
                    f"Error aplicando corrección geométrica:\n{e}"
                )
                return

            # --- 2) MESHS ---
            self._set_busy_message(
                busy,
                "Leyendo meshes de MCNP…\n\nCargando tallies y preparando la interpolación."
            )
            try:
                meshes_all = bu.read_meshtal_all(meshtal_path)
                meshes = {t: meshes_all[t] for t in tallies if t in meshes_all}
                if not meshes:
                    raise RuntimeError("No se cargó ninguna mesh de interés (revisá tallies y archivo).")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Meshtal", f"No se pudo leer la mesh:\n{e}")
                return

            # --- 3) Pipeline: dosis por órgano + export .mat ---
            self._set_busy_message(
                busy,
                "Calculando dosis por órgano…\n\nUniendo segmentación y mesh. Este paso puede tardar."
            )
            try:
                organ_dose = bu.calcular_dosis_por_organo_trilineal(
                    segM=segM,
                    voxel_size_mm=Seg["VoxelSize_mm"],
                    origin_mesh_cm=origin_mesh_cm,
                    meshes=meshes,
                    lut=lut,
                    factor_spnd=factor_spnd
                )

                self._set_busy_message(
                    busy,
                    "Exportando archivos .mat…\n\nGuardando los nuevos vectores de dosis."
                )

                # Normalizar nombres para que matchee RatMaster (VectorDoseRate<Organ>.mat)
                # - quitar espacios / guiones / underscores
                # - remover tildes
                import unicodedata
                import re as _re

                def _norm_key(s: str) -> str:
                    s = str(s).strip()
                    s = unicodedata.normalize("NFKD", s)
                    s = "".join(ch for ch in s if not unicodedata.combining(ch))
                    s = _re.sub(r"[\s_\-]+", "", s)
                    s = _re.sub(r"[^0-9A-Za-z]", "", s)
                    return s

                organ_dose_norm = {_norm_key(k): v for k, v in organ_dose.items()}

                _ = bu.exportar_vectordose_mat(organ_dose_norm, str(out_folder))

                # --- 3b) Guardar datos espaciales para el visor 2D de dosis ---
                # Genera _viz_data.npz + _viz_organ_names.json en la carpeta del set.
                # main_window.reload_vectors() los levanta automáticamente y los pasa
                # a ResultsDialog como viz_data para habilitar el botón "Visualizar Dosis".
                #
                # Se usan los nombres normalizados (igual que los .mat) para que la
                # correspondencia con report["PhysVoxel"] funcione sin conversión extra.
                try:
                    self._set_busy_message(
                        busy,
                        "Guardando datos espaciales para el visor 2D…"
                    )
                    # body_mask_indices: todos los vóxeles con label > 0 en la segmentación
                    body_ijk = np.argwhere(segM > 0).astype(np.int32)

                    # organ_valid_indices: ijk de vóxeles válidos por órgano (nombres normalizados)
                    save_dict = {
                        "seg_shape":         np.array(segM.shape,           dtype=np.int32),
                        "voxel_size_mm":     np.array(Seg["VoxelSize_mm"],  dtype=np.float64),
                        "body_mask_indices": body_ijk,
                    }
                    organ_names_ordered: list[str] = []
                    for raw_organ, data_item in organ_dose.items():
                        norm_name  = _norm_key(raw_organ)
                        orig_idx   = data_item.get("original_indices")
                        valid_mask = data_item.get("valid_mask")
                        if orig_idx is not None and valid_mask is not None:
                            valid_ijk = orig_idx[valid_mask].astype(np.int32)
                            arr_key   = f"organ_{len(organ_names_ordered)}"
                            save_dict[arr_key] = valid_ijk
                            organ_names_ordered.append(norm_name)

                    np.savez_compressed(str(out_folder / "_viz_data.npz"), **save_dict)
                    import json as _json
                    (out_folder / "_viz_organ_names.json").write_text(
                        _json.dumps(organ_names_ordered, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception as _ve:
                    # No es crítico: el resto del pipeline ya terminó bien
                    print(f"[WARN] No se pudo guardar _viz_data.npz: {_ve}")

                # --- 3b) Trazabilidad ---
                try:
                    meta = {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "vector_set": vector_set,
                        "seg_path": str(Path(seg_path).resolve()),
                        "meshtal_path": str(Path(meshtal_path).resolve()),
                        "voxel_size_mm": [float(x) for x in Seg["VoxelSize_mm"]],
                        "origin_mesh_cm": [float(x) for x in origin_mesh_cm],
                        "factor_spnd": float(factor_spnd),
                        "tallies": [int(t) for t in tallies],
                        "tally_overlay": int(tally_overlay),
                        "fix_geom": bool(self.chk_fixgeom.isChecked()),
                    }
                    (out_folder / "_meta.json").write_text(
                        json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                except Exception:
                    pass

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Pipeline",
                    f"No se pudieron generar los vectores:\n{e}"
                )
                return

            # --- 4) Overlay SEG vs MESH (auto-slice) ---
            self._set_busy_message(
                busy,
                "Generando imagen de verificación…\n\nCreando overlay SEG + MESH."
            )
            try:
                self._show_overlay(
                    segM=segM,
                    Seg=Seg,
                    origin_mesh_cm=origin_mesh_cm,
                    meshes=meshes,
                    tally=tally_overlay,
                    bu=bu,
                    save_path=(out_folder / "overlay_check.png")
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Overlay",
                    f"Vectores generados, pero falló el overlay:\n{e}"
                )

            self._result = {
                "seg_path":        seg_path,
                "meshtal_path":    meshtal_path,
                "voxel_size_mm":   tuple(float(x) for x in Seg["VoxelSize_mm"]),
                "origin_mesh_cm":  origin_mesh_cm,
                "factor_spnd":     factor_spnd,
                "tallies":         tallies,
                "tally_overlay":   tally_overlay,
                "out_folder":      str(out_folder),
                "vector_set":      vector_set,
                # Datos espaciales para build_viz_data() en main_window.
                # organ_dose usa nombres normalizados para coincidir con .mat y PhysVoxel.
                "organ_dose":      {_norm_key(k): v for k, v in organ_dose.items()},
                "segM":            segM,
                "report_ref":      {},   # las dosis vienen de compute_bnct; aquí va vacío
            }
            self.accept()

        finally:
            self.btn_run.setEnabled(prev_enabled)
            if busy is not None:
                try:
                    busy.close()
                    busy.deleteLater()
                except Exception:
                    pass

    def _show_overlay(self, segM, Seg, origin_mesh_cm, meshes, tally: int, bu, save_path: Path | None = None):
        import numpy as np
        import matplotlib.pyplot as plt

        if tally not in meshes:
            # si no está, tomar el primero disponible
            tally = list(meshes.keys())[0]

        mesh = meshes[tally]
        cx = mesh["centers"]["x"].copy()
        cy = mesh["centers"]["y"].copy()
        cz = mesh["centers"]["z"].copy()
        M = mesh["Matrix"].copy()

        cx, M = bu._ensure_increasing_axis(cx, M, axis_index=0)
        cy, M = bu._ensure_increasing_axis(cy, M, axis_index=1)
        cz, M = bu._ensure_increasing_axis(cz, M, axis_index=2)

        nx, ny, nz = segM.shape
        xs = bu.seg_phys_coords_1d(nx, Seg["VoxelSize_mm"][0], origin_mesh_cm[0])
        ys = bu.seg_phys_coords_1d(ny, Seg["VoxelSize_mm"][1], origin_mesh_cm[1])
        zs = bu.seg_phys_coords_1d(nz, Seg["VoxelSize_mm"][2], origin_mesh_cm[2])

        counts = np.sum(segM > 0, axis=(0, 1))
        k_seg = int(np.argmax(counts))
        z0 = zs[k_seg]
        k_mesh = int(np.argmin(np.abs(cz - z0)))

        seg_slice = segM[:, :, k_seg]
        mesh_slice = M[:, :, k_mesh]

        body_mask = (seg_slice > 0)
        alpha = np.zeros_like(seg_slice, dtype=float)
        alpha[body_mask] = 0.35

        # extents
        dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
        dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
        x0, x1 = xs[0] - dx / 2, xs[-1] + dx / 2
        y0, y1 = ys[0] - dy / 2, ys[-1] + dy / 2

        mx0, mx1 = bu.mesh_extent_from_centers(cx)
        my0, my1 = bu.mesh_extent_from_centers(cy)

        plt.figure()
        plt.title(
            f"Overlay: MESH {tally} + SEG (alpha)  z≈{z0:.2f} cm  "
            f"(SEG k={k_seg}, MESH k={k_mesh})"
        )
        im = plt.imshow(
            mesh_slice.T,
            origin="lower",
            extent=[mx0, mx1, my0, my1],
            aspect="auto"
        )
        plt.imshow(
            seg_slice.T,
            origin="lower",
            extent=[x0, x1, y0, y1],
            aspect="auto",
            alpha=alpha.T
        )
        plt.contour(xs, ys, body_mask.T.astype(float), levels=[0.5])
        plt.xlabel("x [cm]")
        plt.ylabel("y [cm]")
        bu.safe_colorbar(im, mesh_slice.T, label="tally")
        plt.tight_layout()

        if save_path is not None:
            try:
                plt.savefig(str(save_path), dpi=200)
            except Exception:
                pass

        plt.show()

