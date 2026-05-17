"""
main.py
=======
Punto de entrada de RatMaster.

Responsabilidades (solo orquestación):
  1. Registrar el App User Model ID en Windows (barra de tareas).
  2. Crear la QApplication y aplicar fuente + stylesheet via ui.style.
  3. Mostrar el SplashScreen mientras carga la ventana principal.
  4. Colorear la barra de título nativa de Windows via ui.platform.
  5. Iniciar el event loop de Qt.

Ejecutar con:
    python -m ratmaster.main
  o como ejecutable empaquetado con PyInstaller.
"""

import sys
from pathlib import Path

# ── Garantizar que la carpeta PADRE de ratmaster/ esté en sys.path ────────────
# Permite correr este archivo de dos formas:
#   1. python ratmaster/main.py       (directo, desde la carpeta raíz)
#   2. python -m ratmaster.main       (como módulo, desde la carpeta raíz)
_THIS_FILE   = Path(__file__).resolve()   # …/ratmaster/main.py
_PACKAGE_DIR = _THIS_FILE.parent          # …/ratmaster/
_ROOT_DIR    = _PACKAGE_DIR.parent        # …/  (carpeta que contiene ratmaster/)

if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
# ─────────────────────────────────────────────────────────────────────────────

from PySide6 import QtCore, QtWidgets, QtGui

from ratmaster.app_paths import bundled_icon_path, bundled_resource_path, ensure_user_vectors
from ratmaster.ui.style import apply_app_style, COLOR_TITLEBAR_BG, COLOR_TITLEBAR_TEXT
from ratmaster.ui.platform import apply_custom_titlebar
from ratmaster.ui.splash import SplashScreen
from ratmaster.ui.main_window import BNCTMain


def main() -> None:
    # ── 1. App User Model ID (Windows: agrupa ventanas en la barra de tareas) ──
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "RatMaster.FabioVazquez.1"
        )
    except Exception:
        pass

    # ── 2. QApplication + tema visual ─────────────────────────────────────────
    app = QtWidgets.QApplication(sys.argv)
    apply_app_style(app)

    # ── 3. Ícono de la aplicación ──────────────────────────────────────────────
    icon_path = bundled_icon_path()
    app_icon = None
    if icon_path is not None:
        try:
            app_icon = QtGui.QIcon(str(icon_path))
            if not app_icon.isNull():
                app.setWindowIcon(app_icon)
        except Exception:
            app_icon = None

    # ── 4. SplashScreen ───────────────────────────────────────────────────────
    splash_candidates = [
        bundled_resource_path('assets', 'Ratmaster_logo.png'),
        bundled_resource_path('Ratmaster_logo.png'),
    ]
    splash_path = next((p for p in splash_candidates if p.exists()), None)

    splash: SplashScreen | None = None
    if splash_path is not None:
        pixmap = QtGui.QPixmap(str(splash_path))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                520, 320,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        splash = SplashScreen(pixmap)
        if app_icon is not None and not app_icon.isNull():
            splash.setWindowIcon(app_icon)
        splash.showMessage(
            "  BNCT Dose Analysis Tool\n  Inicializando aplicación...",
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
            QtGui.QColor('white'),
        )
        splash.show()
        QtWidgets.QApplication.processEvents()

    # ── 5. Ventana principal ───────────────────────────────────────────────────
    if splash is not None:
        splash.showMessage(
            "  Cargando módulos y vectores...",
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
            QtGui.QColor('white'),
        )
        QtWidgets.QApplication.processEvents()

    win = BNCTMain()
    if app_icon is not None and not app_icon.isNull():
        win.setWindowIcon(app_icon)

    if splash is not None:
        splash.showMessage(
            "  Listo.",
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
            QtGui.QColor('white'),
        )
        QtWidgets.QApplication.processEvents()

    win.show()

    # ── 6. Barra de título nativa (Windows) ───────────────────────────────────
    # Debe llamarse después de show() para que el HWND esté disponible.
    apply_custom_titlebar(
        hwnd     = int(win.winId()),
        bg_hex   = COLOR_TITLEBAR_BG,    # "#37474F" — mismo que botones
        text_hex = COLOR_TITLEBAR_TEXT,  # "#ECEFF1" — texto claro sobre oscuro
    )

    # ── 7. Cerrar splash y arrancar el loop ───────────────────────────────────
    if splash is not None:
        splash.finish(win)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
