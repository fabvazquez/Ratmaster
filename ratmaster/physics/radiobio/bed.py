"""
physics/radiobio/bed.py
=======================
Dosis Biológicamente Efectiva (BED) y Dosis Equivalente en 2 Gy/fracción (EQD2).

Para la dosis isoefectiva A [Gy fotónico equivalente]:

    BED  = A · (1 + A / (n_fx · (α/β)_R))
    EQD2 = A · ((α/β)_R + A/n_fx) / ((α/β)_R + 2)

donde n_fx es el número de fracciones fotónicas de referencia (escalar).

En BNCT, n_fx = 1 (irradiación única sesión continua); la dosis A ya es el
"equivalente fotónico" calculado por el MLQ con los factores Lea-Catcheside.
Para comparación con radioterapia fraccionada convencional, usar n_fx = 1 y
el cociente α/β del tejido correspondiente.

Nota: EQD2 con n_fx=1 y A en Gy_eq es directamente comparable con planes
de radioterapia fotónica fraccionada en 2 Gy/fx.
"""

from __future__ import annotations
import numpy as np


def bed_voxel(
    A_arr: np.ndarray,
    alpha_beta: float,
    n_fx: int = 1,
) -> np.ndarray:
    """
    BED por vóxel.

    Args:
        A_arr      : dosis isoefectiva por vóxel [Gy_eq].
        alpha_beta : cociente α/β del tejido [Gy].
        n_fx       : número de fracciones (default 1 para BNCT).

    Returns:
        BED por vóxel [Gy].
    """
    A = np.asarray(A_arr, float)
    ab = max(float(alpha_beta), 0.01)
    n  = max(int(n_fx), 1)
    return A * (1.0 + A / (n * ab))


def eqd2_voxel(
    A_arr: np.ndarray,
    alpha_beta: float,
    n_fx: int = 1,
) -> np.ndarray:
    """
    EQD2 (dosis equivalente en 2 Gy/fx) por vóxel.

    Args:
        A_arr      : dosis isoefectiva por vóxel [Gy_eq].
        alpha_beta : cociente α/β del tejido [Gy].
        n_fx       : número de fracciones (default 1 para BNCT).

    Returns:
        EQD2 por vóxel [Gy].
    """
    A = np.asarray(A_arr, float)
    ab = max(float(alpha_beta), 0.01)
    n  = max(int(n_fx), 1)
    return A * (ab + A / n) / (ab + 2.0)


def bed_eqd2_stats(
    A_arr: np.ndarray,
    alpha_beta: float,
    n_fx: int = 1,
) -> dict:
    """
    Estadísticas de BED y EQD2 para un órgano.

    Returns dict:
        BED_mean, BED_max, BED_D50   [Gy]
        EQD2_mean, EQD2_max, EQD2_D50  [Gy]
        alpha_beta_used, n_fx_used
    """
    A = np.asarray(A_arr, float)
    if A.size == 0:
        return {k: 0.0 for k in
                ["BED_mean","BED_max","BED_D50","EQD2_mean","EQD2_max","EQD2_D50"]}

    bed  = bed_voxel(A, alpha_beta, n_fx)
    eq2  = eqd2_voxel(A, alpha_beta, n_fx)

    return {
        "BED_mean":        float(np.mean(bed)),
        "BED_max":         float(np.max(bed)),
        "BED_D50":         float(np.percentile(bed, 50)),
        "EQD2_mean":       float(np.mean(eq2)),
        "EQD2_max":        float(np.max(eq2)),
        "EQD2_D50":        float(np.percentile(eq2, 50)),
        "alpha_beta_used": float(alpha_beta),
        "n_fx_used":       int(n_fx),
    }
