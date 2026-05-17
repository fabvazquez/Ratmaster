"""
main.py
=======
Punto de entrada de RatMaster.

Responsabilidades:
  - Registrar el App User Model ID en Windows (barra de tareas).
  - Crear la QApplication con fuente y stylesheet global.
  - Mostrar el SplashScreen mientras carga la ventana principal.
  - Iniciar el event loop de Qt.

Ejecutar con:
    python -m ratmaster.main
  o como ejecutable empaquetado con PyInstaller.
"""

import sys
import tempfile
from pathlib import Path

# ── Garantizar que la carpeta PADRE de ratmaster/ esté en sys.path ────────────
# Esto permite correr este archivo de dos formas:
#   1. python ratmaster/main.py          (directo, desde la carpeta raíz)
#   2. python -m ratmaster.main          (como módulo, desde la carpeta raíz)
# En ambos casos, la carpeta que contiene a ratmaster/ queda en sys.path.
_THIS_FILE   = Path(__file__).resolve()      # …/ratmaster/main.py
_PACKAGE_DIR = _THIS_FILE.parent             # …/ratmaster/
_ROOT_DIR    = _PACKAGE_DIR.parent           # …/  (carpeta que contiene ratmaster/)

if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
# ─────────────────────────────────────────────────────────────────────────────

from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.app_paths import bundled_icon_path, bundled_resource_path, ensure_user_vectors
from ratmaster.ui.splash import SplashScreen
from ratmaster.ui.main_window import BNCTMain


def _apply_custom_titlebar(hwnd: int,
                            bg_hex: str = "#37474F",
                            text_hex: str = "#FFFFFF") -> None:
    """
    Colorea la barra de título de Windows para que coincida con la barra de menú.

    Usa la API DWM (Desktop Window Manager) de Windows:
      - Windows 11 (build 22000+): color exacto de fondo y texto vía
        DWMWA_CAPTION_COLOR (35) y DWMWA_TEXT_COLOR (36).
      - Windows 10: fallback a modo oscuro del sistema (barra gris oscura)
        vía DWMWA_USE_IMMERSIVE_DARK_MODE (20).
      - Otros SO: no hace nada.

    El hwnd se obtiene con int(win.winId()) después de win.show().
    """
    if sys.platform != "win32":
        return

    def _hex_to_colorref(h: str) -> int:
        """Convierte '#RRGGBB' al formato COLORREF de Windows (0x00BBGGRR)."""
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return r | (g << 8) | (b << 16)

    import ctypes
    dwm = ctypes.windll.dwmapi

    def _set_attr(attr_id: int, value: int) -> bool:
        try:
            v = ctypes.c_uint32(value)
            return dwm.DwmSetWindowAttribute(
                hwnd, attr_id, ctypes.byref(v), ctypes.sizeof(v)
            ) == 0  # S_OK = 0
        except Exception:
            return False

    bg_ref   = _hex_to_colorref(bg_hex)
    text_ref = _hex_to_colorref(text_hex)

    # Windows 11: color exacto de fondo, texto y borde
    ok = _set_attr(35, bg_ref)    # DWMWA_CAPTION_COLOR
    if ok:
        _set_attr(36, text_ref)   # DWMWA_TEXT_COLOR
        _set_attr(34, bg_ref)     # DWMWA_BORDER_COLOR (elimina borde blanco)
    else:
        # Windows 10 fallback: activar modo oscuro del sistema
        _set_attr(20, 1)          # DWMWA_USE_IMMERSIVE_DARK_MODE


def main():
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RatMaster.FabioVazquez.1")
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)

    # Crear archivo SVG temporal para la flecha de ComboBox
    import tempfile as _tempfile
    _arrow_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6"><polygon points="0,0 10,0 5,6" fill="#546E7A"/></svg>'
    _tf = _tempfile.NamedTemporaryFile(suffix='.svg', delete=False, mode='w', encoding='utf-8')
    _tf.write(_arrow_svg)
    _arrow_path = _tf.name.replace('\\', '/')
    _tf.close()
    _arrow_path_disabled = _arrow_path.replace('.svg', '_dis.svg')
    _arrow_svg_dis = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6"><polygon points="0,0 10,0 5,6" fill="#90A4AE"/></svg>'
    try:
        with open(_arrow_path_disabled, 'w', encoding='utf-8') as _tf2:
            _tf2.write(_arrow_svg_dis)
    except Exception:
        _arrow_path_disabled = _arrow_path

    # ── Tipografía base ──────────────────────────────────────────────────
    font = QtGui.QFont("Segoe UI", 10)
    font.setStyleStrategy(QtGui.QFont.PreferAntialias)
    app.setFont(font)

    # ── Stylesheet profesional (Medical/Scientific Software look) ────────
    app.setStyleSheet("""
/* ═══════════════════════════════════════════════════════════════════════
   RatMaster — Tema Gris Oscuro + Menú Claro  (Opción 4)
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
""".replace("__ARROW_PATH__", _arrow_path).replace("__ARROW_PATH_DISABLED__", _arrow_path_disabled))

    icon_path = bundled_icon_path()
    app_icon = None
    if icon_path is not None:
        try:
            app_icon = QtGui.QIcon(str(icon_path))
            if not app_icon.isNull():
                app.setWindowIcon(app_icon)
        except Exception:
            app_icon = None

    splash_candidates = [
        bundled_resource_path('assets', 'Ratmaster_logo.png'),
        bundled_resource_path('Ratmaster_logo.png')
    ]
    splash_path = next((p for p in splash_candidates if p.exists()), None)
    if splash_path is not None:
        pixmap = QtGui.QPixmap(str(splash_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(520, 320, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        splash = SplashScreen(pixmap)
        if app_icon is not None and not app_icon.isNull():
            splash.setWindowIcon(app_icon)
        splash.showMessage("  BNCT Dose Analysis Tool\n  Inicializando aplicación...",
                           QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
                           QtGui.QColor('white'))
        splash.show()
        QtWidgets.QApplication.processEvents()
    else:
        splash = None

    if splash is not None:
        splash.showMessage("  Cargando módulos y vectores...",
                           QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
                           QtGui.QColor('white'))
        QtWidgets.QApplication.processEvents()

    win = BNCTMain()
    if app_icon is not None and not app_icon.isNull():
        win.setWindowIcon(app_icon)

    if splash is not None:
        splash.showMessage("  Listo.",
                           QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
                           QtGui.QColor('white'))
        QtWidgets.QApplication.processEvents()

    win.show()

    # Colorear barra de título para que coincida con la barra de menú (#37474F).
    # Se llama después de show() para garantizar que el HWND esté disponible.
    _apply_custom_titlebar(
        hwnd     = int(win.winId()),
        bg_hex   = "#37474F",   # gris oscuro — mismo que botones
        text_hex = "#ECEFF1",   # gris muy claro sobre oscuro
    )

    if splash is not None:
        splash.finish(win)

    sys.exit(app.exec())
if __name__ == "__main__":
    main()
