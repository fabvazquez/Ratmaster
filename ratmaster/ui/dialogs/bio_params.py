"""
ui/dialogs/bio_params.py
========================
Diálogo para visualizar y editar los parámetros biológicos CBE y RBE
antes de ejecutar el cálculo de dosis equivalente.

CBE (Compound Biological Effectiveness): factor multiplicativo para el boro.
RBE (Relative Biological Effectiveness): factor multiplicativo para neutrones.

Ambas tablas se editan directamente en este diálogo y los cambios
se aplican a la ventana principal al confirmar con "Aceptar".
"""

from PySide6 import QtWidgets

from ratmaster.constants import ORG_ORDER


class BioParamsDialog(QtWidgets.QDialog):
    """
    Muestra las tablas de CBE y RBE del cálculo actual,
    permitiendo editar los valores antes de confirmar.

    La tabla se pre-pobla con los valores actuales de la ventana principal
    (tbl_cbe, tbl_rbe) y los cambios se aplican solo al aceptar.
    """

    def __init__(
        self,
        parent,
        tbl_cbe: QtWidgets.QTableWidget,
        tbl_rbe: QtWidgets.QTableWidget,
    ):
        super().__init__(parent)
        self.setWindowTitle("Parámetros biológicos (CBE / RBE)")
        self.resize(800, 300)

        layout = QtWidgets.QVBoxLayout(self)

        # ── Tabla CBE ────────────────────────────────────────────────────────
        self.tbl_cbe = QtWidgets.QTableWidget()
        self.tbl_cbe.setRowCount(tbl_cbe.rowCount())
        self.tbl_cbe.setColumnCount(tbl_cbe.columnCount())
        self.tbl_cbe.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_cbe.setVerticalHeaderLabels(["CBE"])

        for j in range(tbl_cbe.columnCount()):
            item = tbl_cbe.item(0, j)
            if item:
                self.tbl_cbe.setItem(0, j, QtWidgets.QTableWidgetItem(item.text()))

        # ── Tabla RBE ────────────────────────────────────────────────────────
        self.tbl_rbe = QtWidgets.QTableWidget()
        self.tbl_rbe.setRowCount(tbl_rbe.rowCount())
        self.tbl_rbe.setColumnCount(tbl_rbe.columnCount())
        self.tbl_rbe.setHorizontalHeaderLabels(ORG_ORDER)
        self.tbl_rbe.setVerticalHeaderLabels(["RBE"])

        for j in range(tbl_rbe.columnCount()):
            item = tbl_rbe.item(0, j)
            if item:
                self.tbl_rbe.setItem(0, j, QtWidgets.QTableWidgetItem(item.text()))

        layout.addWidget(QtWidgets.QLabel("CBE (Compound Biological Effectiveness)"))
        layout.addWidget(self.tbl_cbe)
        layout.addWidget(QtWidgets.QLabel("RBE (Relative Biological Effectiveness)"))
        layout.addWidget(self.tbl_rbe)

        # ── Botones ──────────────────────────────────────────────────────────
        btns       = QtWidgets.QHBoxLayout()
        btn_ok     = QtWidgets.QPushButton("Aceptar")
        btn_cancel = QtWidgets.QPushButton("Cancelar")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)
