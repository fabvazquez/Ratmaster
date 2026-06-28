"""
physics/radiobio/tcp_hk.py
==========================
TCP para BNCT con modelo de Hug-Kellerer (HK) + Martel et al. (1999).

Referencia principal:
    Rubén [tesis] Sección 5.2 — modelo de TCP para NSCLC en BNCT.
    González & Santa Cruz (2012) — modelo MLQ / HK para BNCT.

Fundamento
----------
El modelo LQ sobreestima el efecto citotóxico a dosis altas (≥ 6 Gy) porque
la curva de supervivencia sigue curvándose. El modelo Hug-Kellerer (HK)
introduce un tercer parámetro k3 que produce una zona lineal a dosis altas,
acordando con los datos experimentales.

Supervivencia HK (dosis única):
    S_HK(d) = exp(-k1·d + k2·(1 - exp(-k3·d)))

González & Santa Cruz (2012) demostraron que, para n fracciones de dosis d:
    S_HK(D, d) = exp(-D · (k1 - k2·(1-exp(-k3·d))/d))

donde D = n·d es la dosis total.

Efecto biológico por vóxel para irradiación única (d_i = dosis en el vóxel):
    E1i = k1·d_i - k2·(1 - exp(-k3·d_i))

Dosis total equivalente en esquema de 2 Gy/fracción (para usar D50/γ de Martel):
    Di = E1i / (k1 - k2·(1-exp(-k3·2))/2)

TCP total para distribución no uniforme (Martel et al. 1999, Ec. 5.3):
    TCP_total = Π_i  [TCP(Di, 1)]^vi

donde TCP(D) es la función logística de dosis-respuesta:
    TCP(D) = 1 / (1 + (D50/D)^(4γ))

y vi = fracción de volumen del vóxel i (= 1/N para malla uniforme).

Parámetros por defecto
-----------------------
Línea celular H460 (NSCLC) — Park et al. 2008 (ajuste HK de Rubén):
    k1 = 1.81 Gy⁻¹
    k2 = 15.42 Gy⁻¹
    k3 = 0.118 Gy⁻¹

Martel et al. 1999 (NSCLC, 24 meses de seguimiento):
    D50 = 72.0 Gy  (a 2 Gy/fx)
    gamma = 2.0

Incertidumbre 95% CI (para propagar con MC si se desea):
    k1: [1.79, 1.83]
    k2: [14.94, 15.90]
    k3: [0.111, 0.126]
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros por defecto
# ─────────────────────────────────────────────────────────────────────────────

# HK — línea H460 (Park et al. 2008 / ajuste de Rubén)
DEFAULT_K1: float = 1.81    # Gy⁻¹
DEFAULT_K2: float = 15.42   # Gy⁻¹
DEFAULT_K3: float = 0.118   # Gy⁻¹

# IC 95% de los parámetros HK
HK_K1_CI = (1.79, 1.83)
HK_K2_CI = (14.94, 15.90)
HK_K3_CI = (0.111, 0.126)

# Martel et al. 1999 — tabla disponible por intervalo de seguimiento
# {seguimiento_meses: (D50_Gy, gamma, gamma_rango)}
MARTEL_PARAMS: dict[int, tuple] = {
    12: (64.0, 1.3, (0.9, 3.0)),
    24: (72.0, 2.0, (1.0, 4.0)),
    30: (84.5, 1.5, (0.8, 3.0)),
}

DEFAULT_FOLLOWUP_MONTHS: int = 24
DEFAULT_D50:   float = 72.0   # Gy (a 2 Gy/fx)
DEFAULT_GAMMA: float = 2.0    # pendiente normalizada


# ─────────────────────────────────────────────────────────────────────────────
# Modelo HK
# ─────────────────────────────────────────────────────────────────────────────

def survival_hk_single(d_arr: np.ndarray,
                        k1: float = DEFAULT_K1,
                        k2: float = DEFAULT_K2,
                        k3: float = DEFAULT_K3) -> np.ndarray:
    """
    Supervivencia HK para dosis única por vóxel.

    S_HK(d) = exp(-k1·d + k2·(1 - exp(-k3·d)))

    Args:
        d_arr : dosis isoefectiva por vóxel [Gy_eq].
        k1, k2, k3: parámetros HK del modelo de supervivencia.

    Returns:
        S_arr : array de supervivencia ∈ (0, 1] por vóxel.
    """
    d = np.asarray(d_arr, float)
    d = np.clip(d, 0.0, None)
    exponent = k1 * d - k2 * (1.0 - np.exp(-k3 * d))
    return np.exp(-np.clip(exponent, 0.0, 700.0))


def biological_effect_single(d_arr: np.ndarray,
                              k1: float = DEFAULT_K1,
                              k2: float = DEFAULT_K2,
                              k3: float = DEFAULT_K3) -> np.ndarray:
    """
    Efecto biológico E1i para fracción única por vóxel (González & SC 2012).

    E1i = k1·di - k2·(1 - exp(-k3·di))

    Returns:
        E1_arr : efecto biológico [adim] por vóxel.
    """
    d = np.asarray(d_arr, float)
    d = np.clip(d, 0.0, None)
    return k1 * d - k2 * (1.0 - np.exp(-k3 * d))


def eqd2_hk(d_arr: np.ndarray,
             k1: float = DEFAULT_K1,
             k2: float = DEFAULT_K2,
             k3: float = DEFAULT_K3,
             d_ref: float = 2.0) -> np.ndarray:
    """
    Dosis total equivalente en esquema de d_ref Gy/fx (González & SC 2012).

    Di = E1i / (k1 - k2·(1-exp(-k3·d_ref))/d_ref)

    Args:
        d_arr  : dosis isoefectiva BNCT fracción única por vóxel [Gy_eq].
        k1, k2, k3: parámetros HK.
        d_ref  : dosis de referencia por fracción [Gy] (default 2 Gy).

    Returns:
        Di_arr : dosis equivalente por vóxel en el esquema de referencia [Gy].
    """
    E1 = biological_effect_single(d_arr, k1, k2, k3)
    denom = k1 - k2 * (1.0 - np.exp(-k3 * d_ref)) / d_ref
    if abs(denom) < 1e-12:
        return np.zeros_like(E1)
    return np.clip(E1 / denom, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Función logística de dosis-respuesta (Martel et al. 1999)
# ─────────────────────────────────────────────────────────────────────────────

def tcp_logistic_martel(D: float | np.ndarray,
                        D50: float = DEFAULT_D50,
                        gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    """
    TCP logística de Martel et al. (1999) para irradiación uniforme.

    TCP(D) = 1 / (1 + (D50/D)^(4γ))

    Args:
        D    : dosis equivalente [Gy a 2 Gy/fx].
        D50  : dosis para 50% de control tumoral [Gy].
        gamma: pendiente normalizada de la curva sigmoidea.

    Returns:
        TCP ∈ [0, 1].
    """
    D_arr = np.asarray(D, float)
    D_safe = np.where(D_arr > 0, D_arr, 1e-15)
    return 1.0 / (1.0 + (D50 / D_safe) ** (4.0 * gamma))


# ─────────────────────────────────────────────────────────────────────────────
# TCP total para distribución no uniforme (Ec. 5.10 de Rubén)
# ─────────────────────────────────────────────────────────────────────────────

def tcp_hk_total(
    A_arr: np.ndarray,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
    k3: float = DEFAULT_K3,
    D50: float = DEFAULT_D50,
    gamma: float = DEFAULT_GAMMA,
    d_ref: float = 2.0,
    voxel_volumes: np.ndarray | None = None,
) -> float:
    """
    TCP total para distribución de dosis BNCT no uniforme (Ec. 5.10 / 5.3).

    TCP_total = Π_i  [TCP(Di, 1)]^vi

    donde Di es la dosis equivalente en 2 Gy/fx del vóxel i, calculada
    por el modelo HK para fracción única BNCT.

    Args:
        A_arr          : dosis isoefectiva BNCT por vóxel [Gy_eq].
        k1, k2, k3     : parámetros del modelo HK.
        D50, gamma     : parámetros de Martel et al. (1999).
        d_ref          : dosis de referencia por fracción (default 2 Gy).
        voxel_volumes  : volúmenes relativos por vóxel (None → uniforme).

    Returns:
        TCP_total ∈ [0, 1].
    """
    A = np.asarray(A_arr, float)
    if A.size == 0:
        return 0.0

    # Dosis equivalente en esquema d_ref/fracción por vóxel
    Di = eqd2_hk(A, k1, k2, k3, d_ref)

    # TCP logística por vóxel
    tcp_i = tcp_logistic_martel(Di, D50, gamma)

    # Pesos de volumen
    if voxel_volumes is not None:
        v = np.asarray(voxel_volumes, float)
        v = v / v.sum()
    else:
        v = np.ones(A.size) / A.size   # uniforme

    # TCP total = Π_i  tcp_i^vi = exp(Σ_i vi·ln(tcp_i))
    # Proteger contra tcp_i = 0 (log→-inf)
    log_tcp = np.where(tcp_i > 1e-300, np.log(tcp_i), -700.0)
    log_total = float(np.dot(v, log_tcp))
    return float(np.exp(np.clip(log_total, -700.0, 0.0)))


def tcp_hk_stats(
    A_arr: np.ndarray,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
    k3: float = DEFAULT_K3,
    D50: float = DEFAULT_D50,
    gamma: float = DEFAULT_GAMMA,
    d_ref: float = 2.0,
    voxel_volumes: np.ndarray | None = None,
) -> dict:
    """
    TCP HK y estadísticas asociadas.

    Returns dict:
        TCP_HK       — probabilidad de control tumoral [0,1]
        D_eq_mean    — EQD2 media [Gy a d_ref/fx]
        D_eq_D95     — EQD2 percentil D95
        D_iso_mean   — dosis isoefectiva media BNCT [Gy_eq]
        k1, k2, k3   — parámetros HK usados
        D50, gamma   — parámetros Martel usados
        d_ref        — dosis de referencia por fracción usada
        N_voxels     — número de vóxeles
    """
    A = np.asarray(A_arr, float)
    if A.size == 0:
        return {
            "TCP_HK": 0.0, "D_eq_mean": 0.0, "D_eq_D95": 0.0,
            "D_iso_mean": 0.0, "k1": k1, "k2": k2, "k3": k3,
            "D50": D50, "gamma": gamma, "d_ref": d_ref, "N_voxels": 0,
        }

    Di = eqd2_hk(A, k1, k2, k3, d_ref)
    tcp = tcp_hk_total(A, k1, k2, k3, D50, gamma, d_ref, voxel_volumes)

    return {
        "TCP_HK":     tcp,
        "D_eq_mean":  float(np.mean(Di)),
        "D_eq_D95":   float(np.percentile(Di, 5)),   # D95 en DVH = percentil 5
        "D_iso_mean": float(np.mean(A)),
        "k1":         float(k1),
        "k2":         float(k2),
        "k3":         float(k3),
        "D50":        float(D50),
        "gamma":      float(gamma),
        "d_ref":      float(d_ref),
        "N_voxels":   int(A.size),
    }


def tcp_hk_dose_curve(
    dose_Gy: np.ndarray,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
    k3: float = DEFAULT_K3,
    D50: float = DEFAULT_D50,
    gamma: float = DEFAULT_GAMMA,
    d_ref: float = 2.0,
) -> np.ndarray:
    """
    TCP en función de dosis uniforme [Gy_eq BNCT] — curva dosis-respuesta.

    Asume irradiación uniforme (todos los vóxeles reciben la misma dosis).
    Útil para graficar la curva en S y marcar el punto del plan real.

    Returns:
        TCP_arr ∈ [0, 1] para cada dosis en dose_Gy.
    """
    d = np.asarray(dose_Gy, float)
    Di = eqd2_hk(d, k1, k2, k3, d_ref)
    return tcp_logistic_martel(Di, D50, gamma)


def mc_tcp_hk_uncertainty(
    A_arr: np.ndarray,
    sigma_A: np.ndarray,
    k1: float = DEFAULT_K1,
    k2: float = DEFAULT_K2,
    k3: float = DEFAULT_K3,
    D50: float = DEFAULT_D50,
    gamma: float = DEFAULT_GAMMA,
    d_ref: float = 2.0,
    N_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Incertidumbre de TCP_HK por Monte Carlo (perturbando la dosis isoefectiva).

    Args:
        A_arr, sigma_A : dosis y su incertidumbre por vóxel.
        N_samples      : realizaciones MC.

    Returns dict:
        TCP_mean, TCP_std, TCP_p5, TCP_p95, samples
    """
    rng = rng or np.random.default_rng()
    A   = np.asarray(A_arr, float)
    sA  = np.asarray(sigma_A, float)
    if A.size == 0:
        empty = np.array([])
        return {"TCP_mean": 0.0, "TCP_std": 0.0,
                "TCP_p5": 0.0, "TCP_p95": 0.0, "samples": empty}

    noise  = rng.standard_normal((N_samples, A.size))
    A_samp = np.clip(A[None, :] + sA[None, :] * noise, 0.0, None)

    tcp_samples = np.array([
        tcp_hk_total(A_samp[i], k1, k2, k3, D50, gamma, d_ref)
        for i in range(N_samples)
    ])
    return {
        "TCP_mean": float(np.mean(tcp_samples)),
        "TCP_std":  float(np.std(tcp_samples, ddof=1)),
        "TCP_p5":   float(np.percentile(tcp_samples, 5)),
        "TCP_p95":  float(np.percentile(tcp_samples, 95)),
        "samples":  tcp_samples,
    }
