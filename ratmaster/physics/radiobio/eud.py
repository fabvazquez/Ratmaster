"""
physics/radiobio/eud.py
=======================
Dosis Uniforme Equivalente Generalizada (gEUD) — Niemierko 1997.

    gEUD = (Σ_i v_i · d_i^a)^(1/a)

donde v_i = fracción de volumen del vóxel i, d_i = dosis en el vóxel i,
y `a` es el parámetro de volumen:
    a → −∞ : órgano de tipo serie (máxima sensibilidad al punto caliente)
    a = 1  : promedio de dosis
    a → +∞ : órgano de tipo paralelo (sensibilidad al volumen)
    a = −∞ equivale a la dosis mínima, a = +∞ a la máxima.

Se usa con la dosis isoefectiva A (ya en equivalente fotónico).
"""

from __future__ import annotations
import numpy as np


def geud(dose_arr: np.ndarray, a: float) -> float:
    """
    Calcula gEUD de un array de dosis con parámetro de volumen `a`.

    Todos los vóxeles se tratan con igual peso de volumen (uniforme).

    Args:
        dose_arr : dosis por vóxel [Gy o Gy_eq].
        a        : parámetro de volumen (float, puede ser negativo).

    Returns:
        gEUD [mismas unidades que dose_arr]. Retorna 0.0 si array vacío.
    """
    d = np.asarray(dose_arr, float)
    if d.size == 0:
        return 0.0
    d = np.clip(d, 0.0, None)

    # Casos especiales numéricos
    if abs(a) > 1e6:
        return float(np.max(d)) if a > 0 else float(np.min(d))
    if abs(a) < 1e-10:
        # a → 0: gEUD → media geométrica
        log_d = np.where(d > 0, np.log(d), -700.0)
        return float(np.exp(np.mean(log_d)))

    # Caso general: (mean(d^a))^(1/a)
    # Para a < 0 con d=0 → d^a → ∞; usar un clip mínimo
    d_safe = np.where(d > 0, d, 1e-15)
    mean_da = float(np.mean(d_safe ** a))
    if mean_da <= 0:
        return 0.0
    return float(mean_da ** (1.0 / a))


def geud_stats(dose_arr: np.ndarray, a_values: dict[str, float]) -> dict:
    """
    Calcula gEUD para un dict de parámetros `a` nombrados.

    Args:
        dose_arr : dosis por vóxel.
        a_values : {nombre: a} p.ej. {"a_serie": -10, "a_paralelo": 1}.

    Returns:
        dict {nombre: gEUD_value}
    """
    return {name: geud(dose_arr, a) for name, a in a_values.items()}
