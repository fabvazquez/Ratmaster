"""
physics/radiobio/survival.py
============================
Supervivencia celular vóxel a vóxel a partir de la dosis isoefectiva A [Gy_fotón].

La dosis isoefectiva A es ya un "equivalente fotónico" calculado por el modelo MLQ.
La supervivencia celular para fotones (referencia) sigue el modelo LQ:

    S(A) = exp(−α_R · A − G_R · β_R · A²)

donde G_R es el factor Lea-Catcheside de la irradiación FOTÓNICA de referencia
(escalar fijo del preset, no función del tiempo BNCT).

Funciones exportadas:
    survival_voxel(A_arr, p)        → S array por vóxel
    survival_stats(A_arr, p)        → dict {S_mean, S_min, S_max, S_D50}
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


def survival_voxel(A_arr: np.ndarray, aR: float, bR: float, GR: float) -> np.ndarray:
    """
    Supervivencia celular S(A) = exp(−αR·A − GR·βR·A²) por vóxel.

    Args:
        A_arr : array de dosis isoefectiva [Gy fotónico] por vóxel.
        aR    : α del tejido para fotones de referencia [Gy⁻¹].
        bR    : β del tejido para fotones de referencia [Gy⁻²].
        GR    : factor Lea-Catcheside de la referencia fotónica (escalar fijo).

    Returns:
        S_arr : array de supervivencia ∈ (0, 1] por vóxel.
    """
    A = np.asarray(A_arr, float)
    bR_safe = max(float(bR), 0.0)
    exponent = float(aR) * A + float(GR) * bR_safe * A * A
    return np.exp(-np.clip(exponent, 0.0, 700.0))


def survival_stats(A_arr: np.ndarray, aR: float, bR: float, GR: float) -> dict:
    """
    Estadísticas de supervivencia para un órgano.

    Returns dict con:
        S_mean, S_min, S_max   — promedio, mínimo, máximo de S(A)
        S_D50                  — S evaluada en la mediana de A
        N_voxels               — número de vóxeles
    """
    A = np.asarray(A_arr, float)
    if A.size == 0:
        return {"S_mean": 1.0, "S_min": 1.0, "S_max": 1.0,
                "S_D50": 1.0, "N_voxels": 0}
    S = survival_voxel(A, aR, bR, GR)
    return {
        "S_mean":    float(np.mean(S)),
        "S_min":     float(np.min(S)),
        "S_max":     float(np.max(S)),
        "S_D50":     float(survival_voxel(np.array([np.median(A)]), aR, bR, GR)[0]),
        "N_voxels":  int(A.size),
    }
