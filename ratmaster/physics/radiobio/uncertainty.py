"""
physics/radiobio/uncertainty.py
================================
Propagación de incertidumbre en TCP y NTCP mediante Monte Carlo (MC).

La dosis isoefectiva A[v] tiene una incertidumbre σ_A[v] calculada por isoe.py
y almacenada en report["SigmaIsoVoxel"]. Se asume distribución normal por vóxel:

    Ã[v] ~ Normal(A[v], σ_A[v])

Se generan N_samples realizaciones y para cada una se calculan TCP y NTCP.
El resultado es la distribución empírica de TCP/NTCP, de la que se extraen
media y desviación estándar.

Funciones:
    mc_tcp_uncertainty(A_arr, sigma_A, aR, bR, GR, N0, N_samples)
    mc_ntcp_uncertainty(A_arr, sigma_A, TD50, m, n, model, N_samples)
    mc_report(isoe_report, radiobio_params, N_samples)
"""

from __future__ import annotations
import numpy as np
from ratmaster.physics.radiobio.tcp  import tcp_poisson
from ratmaster.physics.radiobio.ntcp import ntcp_lkb, ntcp_logistic

DEFAULT_N_SAMPLES = 500


def mc_tcp_uncertainty(
    A_arr: np.ndarray,
    sigma_A: np.ndarray,
    aR: float,
    bR: float,
    GR: float,
    N0: float,
    N_samples: int = DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Incertidumbre de TCP por Monte Carlo.

    Args:
        A_arr    : dosis isoefectiva por vóxel [Gy_eq].
        sigma_A  : incertidumbre absoluta por vóxel [Gy_eq] (de SigmaIsoVoxel).
        N_samples: número de realizaciones MC.

    Returns dict:
        TCP_mean, TCP_std, TCP_p5, TCP_p95, samples (array)
    """
    rng  = rng or np.random.default_rng()
    A    = np.asarray(A_arr,   float)
    sA   = np.asarray(sigma_A, float)
    if A.size == 0:
        return {"TCP_mean":0.0,"TCP_std":0.0,"TCP_p5":0.0,"TCP_p95":0.0,"samples":np.array([])}

    # Generar realizaciones: (N_samples, N_voxels)
    noise  = rng.standard_normal((N_samples, A.size))
    A_samp = np.clip(A[None, :] + sA[None, :] * noise, 0.0, None)

    tcp_samples = np.array([
        tcp_poisson(A_samp[i], aR, bR, GR, N0)
        for i in range(N_samples)
    ])
    return {
        "TCP_mean": float(np.mean(tcp_samples)),
        "TCP_std":  float(np.std(tcp_samples, ddof=1)),
        "TCP_p5":   float(np.percentile(tcp_samples, 5)),
        "TCP_p95":  float(np.percentile(tcp_samples, 95)),
        "samples":  tcp_samples,
    }


def mc_ntcp_uncertainty(
    A_arr: np.ndarray,
    sigma_A: np.ndarray,
    TD50: float,
    m: float,
    n: float,
    model: str = "lkb",
    N_samples: int = DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Incertidumbre de NTCP por Monte Carlo.

    Returns dict:
        NTCP_mean, NTCP_std, NTCP_p5, NTCP_p95, samples
    """
    rng  = rng or np.random.default_rng()
    A    = np.asarray(A_arr,   float)
    sA   = np.asarray(sigma_A, float)
    if A.size == 0:
        return {"NTCP_mean":0.0,"NTCP_std":0.0,"NTCP_p5":0.0,"NTCP_p95":0.0,"samples":np.array([])}

    fn = ntcp_lkb if model.lower() == "lkb" else ntcp_logistic
    noise    = rng.standard_normal((N_samples, A.size))
    A_samp   = np.clip(A[None, :] + sA[None, :] * noise, 0.0, None)

    ntcp_samples = np.array([
        fn(A_samp[i], TD50, m, n)
        for i in range(N_samples)
    ])
    return {
        "NTCP_mean": float(np.mean(ntcp_samples)),
        "NTCP_std":  float(np.std(ntcp_samples, ddof=1)),
        "NTCP_p5":   float(np.percentile(ntcp_samples,  5)),
        "NTCP_p95":  float(np.percentile(ntcp_samples, 95)),
        "samples":   ntcp_samples,
    }
