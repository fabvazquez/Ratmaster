"""
physics/radiobio/tcp.py
=======================
Control Tumoral — TCP (Tumor Control Probability).

Modelo de Poisson (Webb & Nahum 1993 / Brahme 1984):

    TCP = Π_i P_control(v_i)
        = Π_i exp(−N₀ · v_i · S(A_i))

donde:
    N₀  — número de células clonogénicas en el tumor completo
    v_i — fracción de volumen del vóxel i  (= 1/N_voxels si uniforme)
    S(A_i) — supervivencia celular en el vóxel i (del modelo LQ/MLQ)

Equivalentemente:
    ln(TCP) = −N₀ · Σ_i v_i · S(A_i) = −N₀ · mean(S)

→  TCP = exp(−N₀ · mean(S))

Esta formulación asume que todos los vóxeles tienen el mismo peso de volumen
(malla uniforme). Para mallas no uniformes, proveer `voxel_volumes`.

Funciones:
    tcp_poisson(A_arr, aR, bR, GR, N0)         → float
    tcp_curve(A_scale, A_arr, aR, bR, GR, N0)  → array de TCP vs escala
"""

from __future__ import annotations
import numpy as np
from ratmaster.physics.radiobio.survival import survival_voxel


def tcp_poisson(
    A_arr: np.ndarray,
    aR: float,
    bR: float,
    GR: float,
    N0: float,
    voxel_volumes: np.ndarray | None = None,
) -> float:
    """
    TCP del modelo de Poisson para el órgano tumoral.

    Args:
        A_arr          : dosis isoefectiva por vóxel [Gy_eq].
        aR, bR, GR     : parámetros LQ del tejido tumoral.
        N0             : número de células clonogénicas totales.
        voxel_volumes  : volúmenes relativos por vóxel (si None → uniforme).

    Returns:
        TCP ∈ [0, 1].
    """
    A = np.asarray(A_arr, float)
    if A.size == 0:
        return 0.0

    S = survival_voxel(A, aR, bR, GR)

    if voxel_volumes is not None:
        v = np.asarray(voxel_volumes, float)
        v = v / v.sum()   # normalizar a fracción
        mean_S = float(np.dot(v, S))
    else:
        mean_S = float(np.mean(S))

    # TCP = exp(−N₀ · mean(S))
    exponent = float(N0) * mean_S
    return float(np.exp(-np.clip(exponent, 0.0, 700.0)))


def tcp_curve(
    dose_scales: np.ndarray,
    A_arr: np.ndarray,
    aR: float,
    bR: float,
    GR: float,
    N0: float,
) -> np.ndarray:
    """
    TCP en función de un factor de escala de dosis (para graficar curva S).

    Args:
        dose_scales : array de factores (p.ej. np.linspace(0, 2, 100)).
        A_arr       : dosis isoefectiva base por vóxel.
        aR, bR, GR  : parámetros LQ del tumor.
        N0          : número de células clonogénicas.

    Returns:
        TCP_arr : TCP para cada factor de escala.
    """
    return np.array([
        tcp_poisson(np.asarray(A_arr) * s, aR, bR, GR, N0)
        for s in dose_scales
    ])


def tcp_stats(
    A_arr: np.ndarray,
    aR: float,
    bR: float,
    GR: float,
    N0: float,
    voxel_volumes: np.ndarray | None = None,
) -> dict:
    """
    TCP y estadísticas derivadas para un órgano tumoral.

    Returns dict:
        TCP         — probabilidad de control tumoral [0,1]
        mean_S      — supervivencia media
        N0_used     — N₀ efectivo usado
        D_mean_Gy   — dosis media isoefectiva
        D_D95_Gy    — percentil D95 de la dosis isoefectiva
    """
    A = np.asarray(A_arr, float)
    if A.size == 0:
        return {"TCP": 0.0, "mean_S": 1.0, "N0_used": float(N0),
                "D_mean_Gy": 0.0, "D_D95_Gy": 0.0}

    S = survival_voxel(A, aR, bR, GR)
    if voxel_volumes is not None:
        v = np.asarray(voxel_volumes, float)
        v = v / v.sum()
        mean_S = float(np.dot(v, S))
    else:
        mean_S = float(np.mean(S))

    tcp = float(np.exp(-np.clip(float(N0) * mean_S, 0.0, 700.0)))
    return {
        "TCP":        tcp,
        "mean_S":     mean_S,
        "N0_used":    float(N0),
        "D_mean_Gy":  float(np.mean(A)),
        "D_D95_Gy":   float(np.percentile(A, 5)),  # D95 = percentil 5 del DVH inverso
    }


def tcp_dose_curve(
    dose_Gy: np.ndarray,
    aR: float,
    bR: float,
    GR: float,
    N0: float,
) -> np.ndarray:
    """
    TCP como función de dosis uniforme D [Gy] — curva dosis-respuesta de la literatura.

    Asume que todos los vóxeles reciben exactamente D Gy (irradiación uniforme).
    Esta es la curva teórica en S que se muestra en publicaciones de radiobiología.
    El punto del plan real se marca aparte con la D_media actual.

    Returns:
        Array de TCP ∈ [0,1] para cada dosis en dose_Gy.
    """
    d = np.asarray(dose_Gy, float)
    S = np.exp(-float(aR) * d - float(GR) * float(bR) * d * d)
    return np.exp(-np.clip(float(N0) * S, 0.0, 700.0))