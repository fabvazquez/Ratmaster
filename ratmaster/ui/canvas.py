"""
ui/canvas.py
============
Widget Matplotlib embebido en PySide6 para graficar DVH y otros plots.

MplCanvas encapsula un Figure + Axes con estilo "medical/scientific software":
  - Fondo blanco con grilla gris suave.
  - Spines superior y derecho ocultos.
  - Colores de ejes, ticks y etiquetas en azul-grisáceo oscuro.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtWidgets


class MplCanvas(FigureCanvas):
    """
    Canvas Matplotlib reutilizable para incrustar en layouts de PySide6.

    Uso típico:
        canvas = MplCanvas(parent=self, width=7, height=4, dpi=100)
        layout.addWidget(canvas)
        canvas.ax.plot(x, y, label="Cerebro")
        canvas.ax.legend()
        canvas.draw()
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        width: float = 6,
        height: float = 4,
        dpi: int = 100,
    ):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor("#FFFFFF")

        self.fig = fig
        self.ax  = fig.add_subplot(111)

        # ── Estilo del área de trazado ────────────────────────────────────
        self.ax.set_facecolor("#FAFAFA")

        # Grilla principal y secundaria
        self.ax.grid(True, which="major", color="#E0E0E0", linewidth=0.8, alpha=0.9)
        self.ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.4, alpha=0.7)
        self.ax.minorticks_on()
        self.ax.set_axisbelow(True)   # grilla detrás de las curvas

        # Spines: ocultar superior y derecho; suavizar los restantes
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)
        self.ax.spines["left"].set_color("#BDBDBD")
        self.ax.spines["bottom"].set_color("#BDBDBD")

        # Colores de ejes y etiquetas
        self.ax.tick_params(colors="#546E7A", labelsize=8.5)
        self.ax.xaxis.label.set_color("#37474F")
        self.ax.yaxis.label.set_color("#37474F")
        self.ax.title.set_color("#263238")

        super().__init__(fig)
        self.setParent(parent)
