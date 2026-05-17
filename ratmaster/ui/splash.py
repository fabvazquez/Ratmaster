"""
ui/splash.py
============
Pantalla de carga (splash screen) de RatMaster.

Se muestra mientras la aplicación inicializa los recursos
(copia de vectores, carga de configuración, etc.).
"""

from PySide6 import QtCore, QtWidgets


class SplashScreen(QtWidgets.QSplashScreen):
    """
    Splash screen sin bordes ni barra de título.

    Uso:
        pixmap = QtGui.QPixmap(str(logo_path)).scaled(600, 400, ...)
        splash = SplashScreen(pixmap)
        splash.show()
        app.processEvents()
        # ... inicialización ...
        splash.finish(main_window)
    """

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowOpacity(1)
