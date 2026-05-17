"""
physics/radiobio/models.py
==========================
Punto de entrada único para el análisis radiobiológico completo.

compute_radiobio_report(isoe_report, params_by_organ, isoe_params_by_organ)
    → radiobio_report dict

RadiobioOrganParams — dataclass con los parámetros por órgano para TCP/NTCP/BED.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ratmaster.physics.radiobio.survival   import survival_stats
from ratmaster.physics.radiobio.tcp        import tcp_stats
from ratmaster.physics.radiobio.ntcp       import ntcp_stats
from ratmaster.physics.radiobio.bed        import bed_eqd2_stats
from ratmaster.physics.radiobio.eud        import geud
from ratmaster.physics.radiobio.uncertainty import (
    mc_tcp_uncertainty, mc_ntcp_uncertainty, DEFAULT_N_SAMPLES
)


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros radiobiológicos por órgano
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RadiobioOrganParams:
    """
    Parámetros para el análisis radiobiológico de un órgano.

    Campos:
        organ_type : "tumor" | "oar"
        N0         : células clonogénicas totales (solo tumores)
        TD50       : dosis de tolerancia al 50% [Gy_eq] (solo OAR)
        m          : gradiente de la curva NTCP
        n          : parámetro de volumen para gEUD (a = 1/n)
        alpha_beta : cociente α/β del tejido [Gy] (para BED/EQD2)
        ntcp_model : "lkb" | "logistic"
        run_mc     : si True, calcula incertidumbre MC de TCP/NTCP
    """
    organ_type:  str   = "oar"       # "tumor" | "oar"
    N0:          float = 1e7         # células clonogénicas (tumor)
    TD50:        float = 60.0        # [Gy_eq] (OAR)
    m:           float = 0.15        # gradiente NTCP
    n:           float = 0.1         # parámetro de volumen (a=1/n)
    alpha_beta:  float = 3.0         # [Gy] para BED/EQD2
    ntcp_model:  str   = "lkb"
    run_mc:      bool  = True


# Parámetros por defecto según tipo de tejido
DEFAULT_TUMOR_PARAMS = RadiobioOrganParams(
    organ_type="tumor", N0=1e7, alpha_beta=10.0, ntcp_model="lkb", run_mc=True,
)
DEFAULT_OAR_PARAMS = RadiobioOrganParams(
    organ_type="oar", TD50=60.0, m=0.15, n=0.1, alpha_beta=3.0,
    ntcp_model="lkb", run_mc=True,
)

# Sugerencias predefinidas por tipo de tejido (orientativas)
TISSUE_DEFAULTS: dict[str, RadiobioOrganParams] = {
    "tumor":        RadiobioOrganParams(organ_type="tumor", N0=1e7,  alpha_beta=10.0),
    "normal_brain": RadiobioOrganParams(organ_type="oar",  TD50=72.0, m=0.14, n=0.05, alpha_beta=2.0),
    "spinal_cord":  RadiobioOrganParams(organ_type="oar",  TD50=50.0, m=0.15, n=0.05, alpha_beta=2.0),
    "skin":         RadiobioOrganParams(organ_type="oar",  TD50=55.0, m=0.12, n=0.10, alpha_beta=10.0),
    "mucosa":       RadiobioOrganParams(organ_type="oar",  TD50=65.0, m=0.15, n=0.10, alpha_beta=10.0),
    "lung":         RadiobioOrganParams(organ_type="oar",  TD50=24.5, m=0.18, n=0.87, alpha_beta=3.0),
    "liver":        RadiobioOrganParams(organ_type="oar",  TD50=40.0, m=0.15, n=0.69, alpha_beta=2.5),
    "kidney":       RadiobioOrganParams(organ_type="oar",  TD50=28.0, m=0.12, n=0.70, alpha_beta=2.0),
    "eye":          RadiobioOrganParams(organ_type="oar",  TD50=45.0, m=0.15, n=0.14, alpha_beta=3.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def compute_radiobio_report(
    isoe_report: dict,
    params_by_organ: dict[str, RadiobioOrganParams],
    isoe_params_by_organ: dict,
    N_mc_samples: int = DEFAULT_N_SAMPLES,
    run_mc: bool = True,
) -> dict:
    """
    Calcula el análisis radiobiológico completo a partir del reporte IsoE.

    Args:
        isoe_report         : salida de compute_isoe_from_report (incluye
                              IsoVoxel y SigmaIsoVoxel).
        params_by_organ     : dict {organ_key: RadiobioOrganParams}.
                              Los órganos ausentes se saltan.
        isoe_params_by_organ: dict {organ_key: IsoEParams} — necesario para
                              obtener aR, bR, GR por órgano.
        N_mc_samples        : realizaciones Monte Carlo para incertidumbre.
        run_mc              : si False, omite el MC (más rápido).

    Returns:
        radiobio_report dict con:
            "organs"      : {organ: {TCP|NTCP, BED, EQD2, survival, mc, ...}}
            "summary"     : {TCP_tumor_total, max_NTCP, ...}
            "meta"        : {N_mc_samples, model_notes, ...}
    """
    iso_vox  = isoe_report.get("IsoVoxel",      {})
    sig_vox  = isoe_report.get("SigmaIsoVoxel", {})

    organs_out: dict = {}
    rng = np.random.default_rng(seed=42)

    for organ_key, A_list in iso_vox.items():
        rb_p = params_by_organ.get(organ_key)
        if rb_p is None:
            continue

        iso_p = isoe_params_by_organ.get(organ_key)
        if iso_p is None:
            continue

        A    = np.asarray(A_list, float)
        sA   = np.asarray(sig_vox.get(organ_key, [0.0] * len(A_list)), float)
        if sA.size != A.size:
            sA = np.zeros_like(A)

        aR = iso_p.aR
        bR = iso_p.bR
        GR = iso_p.GR
        ab = rb_p.alpha_beta

        organ_result: dict = {"organ_type": rb_p.organ_type}

        # ── Supervivencia celular ──────────────────────────────────────
        organ_result["survival"] = survival_stats(A, aR, bR, GR)

        # ── BED / EQD2 ────────────────────────────────────────────────
        organ_result["bed_eqd2"] = bed_eqd2_stats(A, ab, n_fx=1)

        # ── TCP (tumores) o NTCP (OAR) ────────────────────────────────
        if rb_p.organ_type == "tumor":
            organ_result["tcp"] = tcp_stats(A, aR, bR, GR, rb_p.N0)
            if run_mc and rb_p.run_mc and sA.any():
                organ_result["mc"] = mc_tcp_uncertainty(
                    A, sA, aR, bR, GR, rb_p.N0, N_mc_samples, rng=rng
                )
        else:
            organ_result["ntcp"] = ntcp_stats(
                A, rb_p.TD50, rb_p.m, rb_p.n, model=rb_p.ntcp_model
            )
            if run_mc and rb_p.run_mc and sA.any():
                organ_result["mc"] = mc_ntcp_uncertainty(
                    A, sA, rb_p.TD50, rb_p.m, rb_p.n,
                    model=rb_p.ntcp_model, N_samples=N_mc_samples, rng=rng
                )

        organs_out[organ_key] = organ_result

    # ── Resumen global ─────────────────────────────────────────────────
    tcp_vals  = [v["tcp"]["TCP"]   for v in organs_out.values()
                 if v.get("organ_type") == "tumor" and "tcp"  in v]
    ntcp_vals = [v["ntcp"]["NTCP"] for v in organs_out.values()
                 if v.get("organ_type") == "oar"   and "ntcp" in v]

    # TCP total = producto de TCPs individuales (independencia de tumores)
    tcp_total = float(np.prod(tcp_vals)) if tcp_vals else None
    max_ntcp  = float(max(ntcp_vals))    if ntcp_vals else None

    summary = {
        "TCP_total":    tcp_total,
        "max_NTCP":     max_ntcp,
        "n_tumors":     len(tcp_vals),
        "n_oar":        len(ntcp_vals),
        "NTCP_by_organ": {
            k: v["ntcp"]["NTCP"]
            for k, v in organs_out.items()
            if "ntcp" in v
        },
        "TCP_by_organ": {
            k: v["tcp"]["TCP"]
            for k, v in organs_out.items()
            if "tcp" in v
        },
    }

    return {
        "organs":  organs_out,
        "summary": summary,
        "meta": {
            "N_mc_samples": N_mc_samples,
            "run_mc":       run_mc,
            "model_notes":  (
                "TCP: modelo de Poisson (Webb & Nahum 1993). "
                "NTCP: LKB (Kutcher & Burman 1989) o logístico. "
                "Dosis isoefectiva A del MLQ (González & Santa Cruz 2012). "
                "BED/EQD2: LQ estándar con n_fx=1 (sesión única BNCT)."
            ),
        },
    }
