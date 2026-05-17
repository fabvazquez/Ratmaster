"""
physics/dose_utils.py
=====================
Utilidades de física de dosis para BNCT:

  - Construcción de DVH (Dose-Volume Histogram) exacto.
  - Cálculo de métricas dosimétricas con incertidumbre (Dmax, Dmin, Dmean, D95, D5).
  - Resumen serializable de matrices de constraints.
  - Cálculo del valor alcanzado para un constraint dado.
  - Utilidades numéricas auxiliares (pad_to_N, percentile_value, numpy_to_list).

Todas las funciones de este módulo son puras (no dependen de Qt ni de IO).
"""

import numpy as np
from ratmaster.constants import ORG_ORDER


# ── Utilidades numéricas ──────────────────────────────────────────────────────

def pad_to_N(a, N: int, default: float) -> np.ndarray:
    """
    Asegura que el array 'a' tenga exactamente N elementos.
    Si es None → llena con 'default'.
    Si es más corto → rellena con 'default'.
    Si es más largo → trunca.
    """
    if a is None:
        return np.full(N, default, dtype=float)
    a = np.array(a, dtype=float).ravel()
    if a.size < N:
        return np.concatenate([a, np.full(N - a.size, default)])
    return a[:N]


def percentile_value(vec: np.ndarray, pct: float) -> float:
    """Valor de dosis en el percentil pct (0..100) del vector de dosis."""
    if vec.size == 0:
        return 0.0
    return float(np.nanpercentile(vec, pct))


def numpy_to_list(obj):
    """
    Convierte recursivamente tipos numpy a tipos nativos de Python,
    para que el resultado sea serializable con json.dumps().
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_list(x) for x in obj]
    return obj


# ── DVH acumulativo ───────────────────────────────────────────────────────────

def build_dvh(doses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye el DVH acumulativo exacto (sin histograma) para un vector de dosis.

    Ordena las dosis de mayor a menor y asigna un porcentaje de volumen lineal.
    El resultado cumple la convención: V(D) = fracción del volumen que recibe ≥ D.

    Returns:
        (dosis_ordenadas_desc, volumen_pct)  ambas de longitud len(doses).
    """
    if doses.size == 0:
        return np.array([]), np.array([])
    s = np.sort(doses)[::-1]
    vol = np.linspace(0, 100, len(s))
    return s, vol


def dvh_extend_to_zero(
    doses_sorted_desc: np.ndarray,
    vol_pct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extiende el DVH hasta D=0 con V=100% si la dosis mínima es mayor que cero.
    Esto asegura la convención estándar del DVH acumulativo.

    Si la última dosis ya es 0, simplemente fuerza V=100 en ese punto.
    """
    D = np.asarray(doses_sorted_desc, dtype=float)
    V = np.asarray(vol_pct, dtype=float)
    if D.size == 0:
        return D, V
    if D[-1] > 0.0:
        D = np.append(D, 0.0)
        V = np.append(V, 100.0)
    else:
        V[-1] = 100.0
    return D, V


# ── Métricas dosimétricas con incertidumbre ───────────────────────────────────

def metrics_with_uncertainty(dvec: np.ndarray, eps_rel: float) -> dict:
    """
    Calcula métricas dosimétricas estándar con incertidumbre absoluta.

    La incertidumbre de cada métrica se estima como valor × eps_rel,
    donde eps_rel es la incertidumbre relativa combinada del cálculo.

    Args:
        dvec:    array de dosis por voxel.
        eps_rel: incertidumbre relativa combinada (fracción, no porcentaje).

    Returns:
        dict con claves: Dmax, Sigma_Dmax, Dmin, Sigma_Dmin,
                         Dmean, Sigma_Dmean, D95, Sigma_D95, D5, Sigma_D5.
    """
    empty = {
        "Dmax": 0.0, "Sigma_Dmax": 0.0,
        "Dmin": 0.0, "Sigma_Dmin": 0.0,
        "Dmean": 0.0, "Sigma_Dmean": 0.0,
        "D95":  0.0, "Sigma_D95":  0.0,
        "D5":   0.0, "Sigma_D5":   0.0,
    }
    if dvec.size == 0:
        return empty

    Dmax  = float(np.nanmax(dvec))
    Dmin  = float(np.nanmin(dvec))
    Dmean = float(np.nanmean(dvec))

    def _percentile_sorted(v: np.ndarray, pct: float) -> float:
        """D_{pct}%: dosis en el pct% más caliente del volumen (convencion DVH)."""
        s = np.sort(v[~np.isnan(v)])[::-1]
        if s.size == 0:
            return 0.0
        idx = max(0, min(s.size - 1, int(round(pct * 0.01 * s.size)) - 1))
        return float(s[idx])

    D95 = _percentile_sorted(dvec, 95)
    D5  = _percentile_sorted(dvec, 5)

    sigma = lambda x: abs(x) * float(eps_rel)
    return {
        "Dmax":  Dmax,  "Sigma_Dmax":  sigma(Dmax),
        "Dmin":  Dmin,  "Sigma_Dmin":  sigma(Dmin),
        "Dmean": Dmean, "Sigma_Dmean": sigma(Dmean),
        "D95":   D95,   "Sigma_D95":   sigma(D95),
        "D5":    D5,    "Sigma_D5":    sigma(D5),
    }


# ── Constraints ───────────────────────────────────────────────────────────────

def summarize_constraints_matrix(
    constraints_matrix: np.ndarray | None,
    organ_order: list | None = None,
) -> list[dict]:
    """
    Convierte una matriz de constraints a una lista de dicts serializable.
    Solo incluye los constraints que tienen valor > 0 (activos).

    Estructura de la matriz (filas):
      0 → Dmax,  1 → Dmean,  2 → Dmin,
      3 → Vx%,   4 → Dose@Vx

    Returns:
        Lista de dicts con claves: org, metric, display, limit_value, vx_pct.
    """
    if constraints_matrix is None:
        return []

    organ_order = list(organ_order or ORG_ORDER)
    try:
        mat = np.asarray(constraints_matrix, dtype=float)
    except Exception:
        return []

    if mat.ndim != 2 or mat.size == 0:
        return []

    items = []
    norg = min(mat.shape[1], len(organ_order))

    for i in range(norg):
        organ = organ_order[i]
        # El pulmón usa 'PulmonTotal' como clave interna
        organ_key = "PulmonTotal" if organ == "Pulmon" else organ

        dmax      = float(mat[0, i]) if mat.shape[0] > 0 else 0.0
        dmean     = float(mat[1, i]) if mat.shape[0] > 1 else 0.0
        dmin      = float(mat[2, i]) if mat.shape[0] > 2 else 0.0
        vx_pct    = float(mat[3, i]) if mat.shape[0] > 3 else 0.0
        dose_at_vx = float(mat[4, i]) if mat.shape[0] > 4 else 0.0

        if dmax > 0:
            items.append({"org": organ_key, "metric": "Dmax",  "display": "Dmax",
                          "limit_value": dmax, "vx_pct": None})
        if dmean > 0:
            items.append({"org": organ_key, "metric": "Dmean", "display": "Dmean",
                          "limit_value": dmean, "vx_pct": None})
        if dmin > 0:
            items.append({"org": organ_key, "metric": "Dmin",  "display": "Dmin",
                          "limit_value": dmin, "vx_pct": None})
        if 0 < vx_pct <= 100 and dose_at_vx > 0:
            items.append({
                "org":         organ_key,
                "metric":      f"D{vx_pct}%",
                "display":     f"Dose@{vx_pct:g}% (Dx)",
                "limit_value": dose_at_vx,
                "vx_pct":      vx_pct,
            })

    return items


def achieved_value_for_constraint(
    voxmap: dict,
    organ_key: str,
    metric: str,
) -> float | None:
    """
    Calcula el valor de dosis alcanzado para un constraint específico.

    Soporta: Dmax, Dmean, Dmin y Dx% (ej: D2%, D50%, D98%).

    Args:
        voxmap:    dict {organ_key: array_de_dosis_por_voxel}
        organ_key: clave del órgano en voxmap
        metric:    nombre del constraint ('Dmax', 'Dmean', 'Dmin', 'D5%', etc.)

    Returns:
        Valor calculado (float) o None si no se puede calcular.
    """
    try:
        if not voxmap or organ_key not in voxmap:
            return None
        vec = np.asarray(voxmap.get(organ_key, []), dtype=float)
        vec = vec[np.isfinite(vec)]
        if vec.size == 0:
            return None

        typ = str(metric or "").strip()
        if typ == "Dmax":
            return float(np.nanmax(vec))
        if typ == "Dmean":
            return float(np.nanmean(vec))
        if typ == "Dmin":
            return float(np.nanmin(vec))

        # Dx%: dosis en el x% más caliente del volumen
        if typ.startswith("D") and typ.endswith("%"):
            try:
                pct = float(typ[1:-1])
            except Exception:
                return None
            if 0.0 < pct <= 100.0:
                return float(np.percentile(vec, 100.0 - pct))

        return None
    except Exception:
        return None
