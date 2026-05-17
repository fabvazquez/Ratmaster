"""
ratmaster
=========
RatMaster — Software de planificación dosimétrica para BNCT (Boron Neutron Capture Therapy).

Estructura del paquete:
    ratmaster/
    ├── main.py              Punto de entrada (función main())
    ├── app_paths.py         Resolución de rutas (dev / PyInstaller)
    ├── constants.py         Constantes globales: órganos, colores, presets
    │
    ├── physics/
    │   ├── dose_utils.py    DVH, métricas, constraints (funciones puras)
    │   ├── compute_bnct.py  Núcleo de cálculo dosimétrico
    │   └── isoe.py          Modelo isoefectivo MLQ (Lea-Catcheside)
    │
    ├── data/
    │   ├── vector_loader.py Carga de .mat con vectores de tasa de dosis
    │   └── persistence.py   Config JSON, registry SPND, parseo numérico
    │
    └── ui/
        ├── formatters.py    Formateo de valores para la UI
        ├── canvas.py        Widget Matplotlib embebido
        ├── splash.py        Pantalla de carga
        ├── main_window.py   Ventana principal (BNCTMain)
        └── dialogs/
            ├── results.py       ResultsDialog + ComponentDosesDialog
            ├── bio_params.py    BioParamsDialog (CBE/RBE)
            ├── boro.py          BoroDialog (protocolos de boro)
            ├── isoe_dialogs.py  IsoEDialog + IsoEPresetsDialog
            ├── constraints.py   ConstraintsDialog + PasteableTable
            ├── spnd.py          SPNDConfigDialog + SPNDFromCurrentsDialog
            └── vector_gen.py    VectorGenDialog (SEG + Meshtal)
"""

__version__ = "12.0"
__author__  = "RatMaster Team"
