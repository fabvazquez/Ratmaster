"""
physics/radiobio/ntcp.py
========================
Probabilidad de complicación en tejido normal — NTCP.

Modelos implementados:

1. LYMAN-KUTCHER-BURMAN (LKB) — Lyman 1985 / Kutcher & Burman 1989
   NTCP = Φ((gEUD − TD50) / (m · TD50))
   donde Φ es la función de distribución acumulada normal estándar.

2. LOGÍSTICO (Niemierko 1997 alternativa)
   NTCP = 1 / (1 + (TD50/gEUD)^(4/m))

Parámetros:
    TD50 — dosis que produce complicación en el 50% de los pacientes [Gy]
    m    — pendiente de la curva (parámetro de gradiente)
    n    — parámetro de volumen (para gEUD: a = 1/n)
    a    — parámetro de volumen de gEUD directamente (a = 1/n)

La gEUD se calcula con el parámetro a = 1/n sobre la dosis isoefectiva A.
"""

from __future__ import annotations
import numpy as np
from scipy.special import ndtr   # CDF normal estándar, más estable que scipy.stats
from ratmaster.physics.radiobio.eud import geud


def ntcp_lkb(
    A_arr: np.ndarray,
    TD50: float,
    m: float,
    n: float,
) -> float:
    """
    NTCP modelo LKB (Lyman-Kutcher-Burman).

    Args:
        A_arr : dosis isoefectiva por vóxel [Gy_eq].
        TD50  : dosis de tolerancia al 50% [Gy_eq].
        m     : parámetro de gradiente de la curva (adim).
        n     : parámetro de volumen (a = 1/n para gEUD).

    Returns:
        NTCP ∈ [0, 1].
    """
    A = np.asarray(A_arr, float)
    if A.size == 0 or TD50 <= 0 or m <= 0:
        return 0.0

    a = 1.0 / max(float(n), 1e-6)
    g = geud(A, a)
    t = (g - float(TD50)) / (float(m) * float(TD50))
    return float(ndtr(t))


def ntcp_logistic(
    A_arr: np.ndarray,
    TD50: float,
    m: float,
    n: float,
) -> float:
    """
    NTCP modelo logístico (Niemierko 1997).

    NTCP = 1 / (1 + (TD50/gEUD)^(4/m))

    Returns NTCP ∈ [0, 1].
    """
    A = np.asarray(A_arr, float)
    if A.size == 0 or TD50 <= 0 or m <= 0:
        return 0.0

    a = 1.0 / max(float(n), 1e-6)
    g = geud(A, a)
    if g <= 0:
        return 0.0

    ratio = float(TD50) / g
    return float(1.0 / (1.0 + ratio ** (4.0 / float(m))))


def ntcp_stats(
    A_arr: np.ndarray,
    TD50: float,
    m: float,
    n: float,
    model: str = "lkb",
) -> dict:
    """
    NTCP y estadísticas asociadas para un órgano en riesgo.

    Args:
        A_arr  : dosis isoefectiva por vóxel.
        TD50   : dosis de tolerancia al 50% [Gy_eq].
        m      : gradiente.
        n      : parámetro de volumen.
        model  : "lkb" | "logistic".

    Returns dict:
        NTCP         — probabilidad de complicación [0,1]
        gEUD_Gy      — gEUD del órgano [Gy_eq]
        D_mean_Gy    — dosis media del órgano
        D_max_Gy     — dosis máxima del órgano
        model_used   — nombre del modelo
        TD50, m, n   — parámetros utilizados
    """
    A = np.asarray(A_arr, float)
    a = 1.0 / max(float(n), 1e-6)
    g = geud(A, a) if A.size > 0 else 0.0

    fn = ntcp_lkb if model.lower() == "lkb" else ntcp_logistic
    ntcp = fn(A, TD50, m, n)

    return {
        "NTCP":       ntcp,
        "gEUD_Gy":    g,
        "D_mean_Gy":  float(np.mean(A)) if A.size > 0 else 0.0,
        "D_max_Gy":   float(np.max(A))  if A.size > 0 else 0.0,
        "model_used": model,
        "TD50":       float(TD50),
        "m":          float(m),
        "n":          float(n),
    }


def ntcp_curve(
    dose_scales: np.ndarray,
    A_arr: np.ndarray,
    TD50: float,
    m: float,
    n: float,
    model: str = "lkb",
) -> np.ndarray:
    """
    NTCP en función de un factor de escala de dosis (para graficar).

    Returns array de NTCP para cada factor en dose_scales.
    """
    fn = ntcp_lkb if model.lower() == "lkb" else ntcp_logistic
    return np.array([
        fn(np.asarray(A_arr) * s, TD50, m, n)
        for s in dose_scales
    ])
