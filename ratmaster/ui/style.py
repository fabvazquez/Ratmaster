"""
ui/style.py
===========
Tipografía, stylesheet QSS y recursos visuales de RatMaster.

Exporta:
    apply_app_style(app) → None
        Configura la fuente base y aplica el stylesheet completo a la QApplication.
        Crea los SVGs temporales de la flecha del ComboBox al vuelo.

No depende de ningún otro módulo de ratmaster, solo de PySide6 y stdlib.
"""

import tempfile
from PySide6 import QtGui, QtWidgets


# ── Constantes de color del tema ──────────────────────────────────────────────
# Se definen aquí para que platform.py (y cualquier otro módulo) pueda
# importarlas sin duplicar literales.
COLOR_TITLEBAR_BG   = "#37474F"   # gris oscuro — botones y barra de título DWM
COLOR_TITLEBAR_TEXT = "#ECEFF1"   # gris muy claro — texto sobre fondo oscuro
COLOR_MENUBAR_BG    = "#ECEFF1"   # gris claro — barra de menú
COLOR_STATUSBAR_BG  = "#263238"   # gris muy oscuro — barra de estado


# ── Template del stylesheet ───────────────────────────────────────────────────
# Los marcadores __ARROW_PATH__ y __ARROW_PATH_DISABLED__ se reemplazan en
# tiempo de ejecución por apply_app_style() con las rutas reales de los SVGs.

_QSS_TEMPLATE = """
/* ═══════════════════════════════════════════════════════════════════════
   RatMaster — Tema Gris Oscuro + Menú Claro
   #37474F  Gris oscuro   — barra título (DWM), botones
   #ECEFF1  Gris claro    — barra de menú
   #263238  Gris muy osc. — barra de estado (cierra el marco)
   #FAFAFA  Casi blanco   — fondo general
   #FFFFFF  Blanco        — groupbox, inputs, tablas
   #455A64  Medio         — hover botones, títulos de grupo
   #546E7A  Medio-claro   — texto secundario, bordes
═══════════════════════════════════════════════════════════════════════ */

/* ── Base ─────────────────────────────────────────────────────────────── */
QWidget {
    font-family: "Segoe UI", "SF Pro Text", "Ubuntu", Arial, sans-serif;
    font-size: 10pt;
    color: #263238;
    background-color: #FAFAFA;
}
QMainWindow { background-color: #FAFAFA; }
QDialog     { background-color: #FAFAFA; }

/* ── Barra de menú ────────────────────────────────────────────────────── */
QMenuBar {
    background-color: #ECEFF1;
    color: #37474F;
    border-bottom: 1px solid #CFD8DC;
    spacing: 2px;
    padding: 2px 6px;
    font-weight: 600;
}
QMenuBar::item {
    background: transparent;
    padding: 5px 12px;
    border-radius: 3px;
    color: #37474F;
}
QMenuBar::item:selected  { background-color: #CFD8DC; color: #263238; }
QMenuBar::item:pressed   { background-color: #B0BEC5; color: #263238; }
QMenu {
    background-color: #FFFFFF;
    border: 1px solid #CFD8DC;
    border-radius: 5px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 14px;
    border-radius: 3px;
    color: #263238;
}
QMenu::item:selected { background-color: #ECEFF1; color: #263238; }
QMenu::separator     { height: 1px; background: #E0E0E0; margin: 4px 8px; }

/* ── GroupBox ─────────────────────────────────────────────────────────── */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #CFD8DC;
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: 700;
    font-size: 9.5pt;
    color: #455A64;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -1px;
    padding: 0 6px;
    color: #455A64;
    background-color: #FFFFFF;
}

/* ── Botones ──────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #37474F;
    color: #ECEFF1;
    border: none;
    border-radius: 4px;
    padding: 5px 18px;
    font-weight: 600;
    font-size: 9.5pt;
    min-height: 26px;
}
QPushButton:hover   { background-color: #455A64; }
QPushButton:pressed { background-color: #263238; }
QPushButton:disabled {
    background-color: #CFD8DC;
    color: #90A4AE;
}
QDialogButtonBox QPushButton { min-width: 80px; }

/* ── Entradas de texto ────────────────────────────────────────────────── */
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #B0BEC5;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #B2DFDB;
    color: #263238;
}
QLineEdit:focus     { border-color: #455A64; border-width: 1.5px; }
QLineEdit:read-only { background-color: #ECEFF1; color: #546E7A; }

/* ── ComboBox ─────────────────────────────────────────────────────────── */
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #B0BEC5;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    color: #263238;
    selection-background-color: #B2DFDB;
}
QComboBox:hover { border-color: #546E7A; }
QComboBox:focus { border-color: #455A64; }
QComboBox::drop-down {
    border: none;
    border-left: 1px solid #CFD8DC;
    width: 26px;
    background: #ECEFF1;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}
QComboBox::down-arrow {
    image: url(__ARROW_PATH__);
    width: 10px;
    height: 6px;
}
QComboBox::down-arrow:disabled {
    image: url(__ARROW_PATH_DISABLED__);
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #CFD8DC;
    selection-background-color: #ECEFF1;
    selection-color: #263238;
    outline: none;
}

/* ── RadioButton / CheckBox ───────────────────────────────────────────── */
QRadioButton { spacing: 7px; background: transparent; }
QRadioButton::indicator {
    width: 15px; height: 15px;
    border: 2px solid #90A4AE;
    border-radius: 8px;
    background: #FFFFFF;
}
QRadioButton::indicator:checked {
    border-color: #455A64;
    background-color: #455A64;
}
QCheckBox { spacing: 7px; background: transparent; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 2px solid #90A4AE;
    border-radius: 3px;
    background: #FFFFFF;
}
QCheckBox::indicator:checked {
    border-color: #455A64;
    background-color: #455A64;
}

/* ── Tablas ───────────────────────────────────────────────────────────── */
QTableWidget, QTableView {
    background-color: #FFFFFF;
    alternate-background-color: #F5F7FA;
    border: 1px solid #CFD8DC;
    border-radius: 4px;
    gridline-color: #E8EAED;
    selection-background-color: #CFD8DC;
    selection-color: #263238;
    outline: none;
}
QTableWidget::item { padding: 3px 7px; }
QHeaderView::section {
    background-color: #ECEFF1;
    border: none;
    border-right:  1px solid #CFD8DC;
    border-bottom: 1px solid #CFD8DC;
    padding: 5px 8px;
    font-weight: 700;
    font-size: 9pt;
    color: #455A64;
}
QHeaderView::section:last { border-right: none; }

/* ── Scrollbars ───────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #ECEFF1; width: 10px; border-radius: 5px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #B0BEC5; border-radius: 5px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #90A4AE; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #ECEFF1; height: 10px; border-radius: 5px; margin: 0;
}
QScrollBar::handle:horizontal {
    background: #B0BEC5; border-radius: 5px; min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #90A4AE; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Splitter ─────────────────────────────────────────────────────────── */
QSplitter::handle { background-color: #CFD8DC; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical   { height: 2px; }

/* ── Labels ───────────────────────────────────────────────────────────── */
QLabel { background: transparent; color: #263238; }

/* ── ListWidget ───────────────────────────────────────────────────────── */
QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #CFD8DC;
    border-radius: 4px;
    outline: none;
}
QListWidget::item { padding: 4px 6px; }
QListWidget::item:selected {
    background-color: #ECEFF1;
    color: #263238;
}

/* ── Barra de estado ──────────────────────────────────────────────────── */
QStatusBar {
    background-color: #263238;
    color: #B0BEC5;
    font-size: 9pt;
}
QStatusBar::item { border: none; }

/* ── ToolTip ──────────────────────────────────────────────────────────── */
QToolTip {
    background-color: #455A64;
    color: #ECEFF1;
    border: none;
    padding: 5px 9px;
    border-radius: 4px;
    font-size: 9pt;
}
"""


def _create_combo_arrow_svgs() -> tuple[str, str]:
    """
    Crea dos SVGs temporales para la flecha del QComboBox:
      - flecha normal  (#546E7A)
      - flecha disabled (#90A4AE)

    Devuelve (arrow_path, arrow_disabled_path) con barras forward
    para que el QSS las acepte en cualquier plataforma.
    """
    arrow_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
        '<polygon points="0,0 10,0 5,6" fill="#546E7A"/></svg>'
    )
    arrow_svg_dis = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
        '<polygon points="0,0 10,0 5,6" fill="#90A4AE"/></svg>'
    )

    tf = tempfile.NamedTemporaryFile(suffix='.svg', delete=False, mode='w', encoding='utf-8')
    tf.write(arrow_svg)
    arrow_path = tf.name.replace('\\', '/')
    tf.close()

    arrow_path_disabled = arrow_path.replace('.svg', '_dis.svg')
    try:
        with open(arrow_path_disabled, 'w', encoding='utf-8') as f:
            f.write(arrow_svg_dis)
    except Exception:
        arrow_path_disabled = arrow_path   # fallback: usar la misma flecha

    return arrow_path, arrow_path_disabled


def apply_app_style(app: QtWidgets.QApplication) -> None:
    """
    Aplica la fuente base y el stylesheet completo a *app*.

    Llamar una sola vez, antes de crear cualquier widget.
    """
    # Fuente base
    font = QtGui.QFont("Segoe UI", 10)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(font)

    # SVGs de la flecha del ComboBox
    arrow_path, arrow_path_disabled = _create_combo_arrow_svgs()

    # Stylesheet con rutas resueltas
    qss = (
        _QSS_TEMPLATE
        .replace("__ARROW_PATH__", arrow_path)
        .replace("__ARROW_PATH_DISABLED__", arrow_path_disabled)
    )
    app.setStyleSheet(qss)
