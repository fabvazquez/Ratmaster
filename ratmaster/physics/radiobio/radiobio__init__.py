"""
physics/radiobio/
=================
Módulo de análisis radiobiológico para BNCT.

Calcula probabilidades de control tumoral (TCP) y de complicación en tejido normal
(NTCP) a partir de la distribución de dosis isoefectiva generada por isoe.py.

Módulos:
    survival   — supervivencia celular S_mix(A) por vóxel (modelo MLQ)
    tcp        — TCP por órgano (Poisson + integración DVH)
    ntcp       — NTCP por órgano (Lyman-Kutcher-Burman y gEUD)
    eud        — Dosis Uniforme Equivalente Generalizada (gEUD)
    bed        — BED y EQD2 por órgano
    uncertainty — propagación Monte Carlo de incertidumbres en TCP/NTCP
    models     — punto de entrada único: compute_radiobio_report()
"""
from ratmaster.physics.radiobio.models import (
    compute_radiobio_report,
    RadiobioOrganParams,
)

__all__ = ["compute_radiobio_report", "RadiobioOrganParams"]
