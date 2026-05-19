"""
physics/compute_bnct.py
=======================
Núcleo de cálculo dosimétrico BNCT.

La función principal `compute_bnct()` realiza tres pasos:

  PASO 1 — Determinación del tiempo de irradiación.
    En modo "tiempo fijo": usa el tiempo ingresado por el usuario.
    En modo "constraints": calcula el tiempo máximo permitido por cada constraint
    (Dmax, Dmean, Dmin, Dose@Vx%) y elige el mínimo (constraint más restrictivo).

  PASO 2 — Cálculo de dosis por vóxel.
    Para cada órgano y cada componente (Boro, Rápidos, Térmicos, Gamma):

        D_i[v] = r_i[v] [Gy·cm²/n] × spnd_value [n/(cm²·s)] / FLUX_CORR_FACTOR × t [s]

    Los vectores de tasas r_i[v] ya tienen unidades [Gy·cm²/n] porque en la carga
    del meshtal se dividió la tasa MCNP [Gy/partícula fuente] por el factor SPND
    simulado [n/(cm²·partícula fuente)], cancelando la unidad "partícula fuente".
    El cociente (spnd_value / FLUX_CORR_FACTOR) tiene unidades [n/(cm²·s)] y
    representa el flujo efectivo en el detector durante la irradiación.

    FLUX_CORR_FACTOR es una constante adimensional que corrige la discrepancia entre
    el flujo simulado y el flujo real (no es un valor del SPND de referencia).

  PASO 3 — Identificación del constraint limitante.
    Calcula qué constraint se satura primero y lo registra en el reporte.

MODELO DE INCERTIDUMBRE (propagación vóxel a vóxel):
----------------------------------------------------

MODO TIEMPO FIJO (t medido independientemente):
    D_i[v] = r_i[v] × spnd_value/FLUX_CORR_FACTOR × t
    Todas las fuentes son independientes entre sí y propagan normalmente.

MODO CONSTRAINTS (t derivado del constraint):
    t = D_lim / (r_binding × spnd_value/FLUX_CORR_FACTOR)
    ⟹ D[v] = D_lim × r_mcnp[v] / r_mcnp_binding

    El factor (spnd_value/FLUX_CORR_FACTOR) CANCELA ALGEBRAICAMENTE.
    Por tanto eps_S y eps_time NO propagan a las dosis de vóxel.

ESTRUCTURA DE CORRELACIONES (ambos modos):
    eps_S, eps_time y eps_sys afectan a TODOS los componentes por igual
    a través del mismo factor global (spnd_value/FLUX_CORR_FACTOR × t).
    Los componentes D_b, D_f, D_t, D_g están correlacionados entre sí
    por esas fuentes. La propagación correcta de varianza es:

        σ²(D_phys) = σ²_indep(D_b) + σ²_indep(D_f) + σ²_indep(D_t) + σ²_indep(D_g)
                   + (eps_S_dose² + eps_time_dose² + eps_sys²) × D_phys²

    donde el último término captura los cross-terms de la covarianza entre
    componentes (el factor D_phys² surge de Cauchy-Schwarz sobre la suma).
    La fórmula ingenua con Σ D_i² en vez de D_phys² omite esos cross-terms
    y subestima σ cuando varios componentes son comparables.

    Fuentes independientes por componente:
        eps_B    — incertidumbre en [B] → SOLO afecta D_b
        eps_mcnp[v] — estadística MCNP por vóxel → afecta cada componente independientemente

    Fuentes correlacionadas (valor efectivo según modo):
        eps_S_dose    = eps_S  (modo fijo) | 0.0 (modo constraints, SPND cancela)
        eps_time_dose = eps_time (modo fijo) | 0.0 (modo constraints, t derivado)
        eps_sys       = error sistemático adicional (igual en ambos modos)

Para la dosis biológica H = CBE×D_b + RBE×(D_f+D_t) + D_g:
    misma estructura, usando H² para el término correlacionado.

El resultado es un dict `Report` con:
    - "meta":            metadatos del cálculo (tiempo, modo, constraints usados)
    - "CompDoses":       componentes por vóxel (Boro, Fstn, Thn, Gamma) × órgano
                         + errores absolutos por componente (sigma_Boro, etc.)
    - "PhysVoxel":       dosis física total por vóxel × órgano
    - "BioVoxel":        dosis equivalente (RBE/CBE) por vóxel × órgano
    - "SigmaPhysVoxel":  error absoluto de dosis física por vóxel × órgano
    - "SigmaBioVoxel":   error absoluto de dosis biológica por vóxel × órgano
    - "ParamsUsed":      B, B_err, CBE, RBE utilizados
"""

from datetime import datetime, timezone
import numpy as np

from ratmaster.constants import FLUX_CORR_FACTOR
from ratmaster.physics.dose_utils import (  
    pad_to_N,
    metrics_with_uncertainty,
    summarize_constraints_matrix,
)


def compute_bnct(
    vectors: dict,
    organ_order: list,
    B_arr,
    B_err_arr,
    CBE_arr,
    RBE_arr,
    spnd_value: float,
    time_s: float,
    time_err: float,
    mode_constraints: bool,
    constraints_matrix,
    vectors_err: dict | None = None,
    sys_error: float = 0.0,
    spnd_error: float = 0.0,
    dose_for_limits: str = "phys",
    time_from_constraints: bool = False,
) -> tuple[dict, dict, dict, dict]:
    """
    Calcula la distribución de dosis BNCT por vóxel para todos los órganos.

    Args:
        vectors:           dict {organ_key: (Boro, Fstn, Thn, Gamma)} con tasas
                           de dosis por vóxel [Gy·cm²/n].
                           Las tasas ya fueron normalizadas por el factor SPND
                           simulado: r = tally[Gy/part.] / factor_spnd[n/(cm²·part.)].
        organ_order:       lista de nombres lógicos de órganos (ej: ORG_ORDER).
        B_arr:             concentración de boro [ppm] por órgano.
        B_err_arr:         incertidumbre ABSOLUTA en concentración de boro [ppm].
        CBE_arr:           Compound Biological Effectiveness por órgano.
        RBE_arr:           Relative Biological Effectiveness por órgano.
        spnd_value:        lectura del SPND durante la irradiación [n/(cm²·s)].
        time_s:            tiempo de irradiación [s] (usado en modo tiempo fijo).
        time_err:          incertidumbre ABSOLUTA del tiempo [s] (modo tiempo fijo).
        mode_constraints:  si True, calcula el tiempo desde los constraints.
        constraints_matrix: np.ndarray (5 × N_organs) o None.
        vectors_err:       dict {organ_key: (Br_err, Fr_err, Tr_err, Gr_err)} con
                           errores relativos MCNP por vóxel para cada componente.
                           Si es None o falta una clave, se asume error MCNP = 0.
        sys_error:         error sistemático relativo adicional (fracción, adim.).
        spnd_error:        incertidumbre ABSOLUTA del SPND [n/(cm²·s)].
                           La incertidumbre relativa eps_S = spnd_error / spnd_value
                           se calcula internamente. No pasar la fracción relativa
                           pre-calculada (eso era el bug anterior).
        dose_for_limits:   "phys" o "bio" — modo de dosis para evaluar constraints.
        time_from_constraints: si True, el tiempo fue determinado externamente por
                           un sistema de constraints (ej: solve_time_isoe_from_constraints).
                           Tiene el mismo efecto que mode_constraints=True sobre las
                           incertidumbres: eps_S y eps_time se excluyen de las varianzas
                           de dosis porque el factor SPND cancela en la ratio
                           D[v] = r[v] × t = D_lim × r_mcnp[v] / r_mcnp_binding.

    Returns:
        (Report, Dsum_phys, Dsum_bio, Deq)
        - Report:     dict con toda la información del cálculo (ver descripción del módulo).
        - Dsum_phys:  {organ_key: dict de métricas físicas con incertidumbre}.
        - Dsum_bio:   {organ_key: dict de métricas equivalentes con incertidumbre}.
        - Deq:        {organ_key: dict con H_mean, H_max, sigma, CBE, RBE, B_used, B_err}.
    """
    if vectors_err is None:
        vectors_err = {}

    N = len(organ_order)
    B_arr     = pad_to_N(B_arr,     N, 0.0)
    B_err_arr = pad_to_N(B_err_arr, N, 0.0)
    CBE_arr   = pad_to_N(CBE_arr,   N, 1.0)
    RBE_arr   = pad_to_N(RBE_arr,   N, 1.0)

    # Factor de corrección de flujo:
    # Corr [n/(cm²·s)] = spnd_value / FLUX_CORR_FACTOR
    # Multiplicado por las tasas r_i [Gy·cm²/n] da la tasa de dosis [Gy/s].
    # FLUX_CORR_FACTOR es adimensional y corrige la discrepancia entre la
    # simulación MCNP y la realidad (no es un valor del SPND de referencia).
    Corr = float(spnd_value) / float(FLUX_CORR_FACTOR)

    # ── Incertidumbres globales (escalares, mismas para todos los vóxeles) ───
    #
    # eps_S: incertidumbre RELATIVA del SPND = sigma_spnd / spnd_value.
    # spnd_error se recibe como valor ABSOLUTO [n/(cm²·s)].
    eps_S = (
        abs(float(spnd_error)) / abs(float(spnd_value))
        if float(spnd_value) != 0
        else 0.0
    )

    # eps_time: incertidumbre RELATIVA del tiempo = sigma_t / t.
    # Solo relevante en modo tiempo fijo; en modo constraints, t es derivado.
    eps_time = (
        abs(float(time_err)) / float(time_s)
        if (float(time_s) > 0 and float(time_err) > 0)
        else 0.0
    )

    # eps_sys: error sistemático adicional (fracción relativa) Actualmente es 0 por default.
    eps_sys = abs(float(sys_error)) if sys_error else 0.0

    # ── Función interna: tasa de dosis por vóxel ────────────────────────────

    def dose_rate_for_orgidx(i: int, key: str, mode: str) -> np.ndarray:
        """
        Calcula la tasa de dosis total para un órgano dado.
        mode='phys' → suma lineal de componentes.
        mode='bio'  → pesada por CBE (Boro) y RBE (neutrones).
        """
        Br, Fr, Tr, Gr = vectors[key]
        Bval = float(B_arr[i])
        CBE  = float(CBE_arr[i])
        RBE  = float(RBE_arr[i])

        r_b = Br * Bval * Corr  # tasa boro
        r_f = Fr * Corr          # tasa neutrones rápidos
        r_t = Tr * Corr          # tasa neutrones térmicos
        r_g = Gr * Corr          # tasa gamma

        if mode == "bio":
            return CBE * r_b + RBE * (r_f + r_t) + r_g
        return r_b + r_f + r_t + r_g

    # ════════════════════════════════════════════════════════════════════════
    # PASO 1 — Determinación del tiempo de irradiación
    # ════════════════════════════════════════════════════════════════════════

    _dmin_conflict_info = None

    if mode_constraints and constraints_matrix is not None:
        # Calcula el tiempo que satisface cada constraint y elige el mínimo
        candidate_times: list[float] = []
        candidate_tags:  list[tuple] = []
        rows = constraints_matrix.shape[0]

        for i, organ_logic in enumerate(organ_order):
            # El Pulmón usa clave compuesta 'PulmonTotal' para el cálculo
            key = "PulmonTotal" if organ_logic == "Pulmon" else organ_logic
            if key not in vectors:
                continue

            rate = dose_rate_for_orgidx(i, key, dose_for_limits)
            if rate.size == 0:
                continue

            Dmax_allowed  = float(constraints_matrix[0, i]) if rows > 0 else 0.0
            Dmean_allowed = float(constraints_matrix[1, i]) if rows > 1 else 0.0
            Dmin_allowed  = float(constraints_matrix[2, i]) if rows > 2 else 0.0
            Vx_pct        = float(constraints_matrix[3, i]) if rows > 3 else 0.0
            Dose_at_Vx    = float(constraints_matrix[4, i]) if rows > 4 else 0.0

            # Tiempo para Dmax: D = rate_max × t → t = Dmax / rate_max
            if Dmax_allowed > 0:
                rmax = np.nanmax(rate)
                if rmax > 0:
                    candidate_times.append(Dmax_allowed / rmax)
                    candidate_tags.append((key, "Dmax", Dmax_allowed, i))

            # Tiempo para Dmean: Dmean = rate_mean × t
            if Dmean_allowed > 0:
                rmean = np.nanmean(rate)
                if rmean > 0:
                    candidate_times.append(Dmean_allowed / rmean)
                    candidate_tags.append((key, "Dmean", Dmean_allowed, i))

            # Tiempo para Dose@Vx%: dosis en el Vx% más caliente
            if Vx_pct > 0 and Dose_at_Vx > 0 and len(rate) > 0:
                rvx = np.percentile(rate, 100.0 - Vx_pct)
                if rvx > 0:
                    candidate_times.append(Dose_at_Vx / rvx)
                    candidate_tags.append((key, f"D{Vx_pct}%", Dose_at_Vx, i))

            # Tiempo para Dmin: Dmin = rate_min × t (constraint de cobertura tumoral)
            if Dmin_allowed > 0:
                rmin = np.nanmin(rate)
                if rmin > 0:
                    candidate_times.append(Dmin_allowed / rmin)
                    candidate_tags.append((key, "Dmin", Dmin_allowed, i))

        if not candidate_times:
            raise RuntimeError(
                "No se generaron tiempos candidatos a partir de los constraints. "
                "Verificá que los vectores de dosis estén cargados correctamente."
            )

        time_used = float(min(candidate_times))

    else:
        # Modo tiempo fijo: usar el tiempo ingresado por el usuario
        time_used = float(time_s)

    # ── Routing de incertidumbres según origen del tiempo ───────────────────
    #
    # MODO CONSTRAINTS (mode_constraints=True ó time_from_constraints=True):
    #
    #   t = D_lim / (r_binding × spnd_value / FLUX_CORR_FACTOR)
    #   D[v] = r[v] × (spnd_value/FLUX_CORR_FACTOR) × t
    #        = D_lim × r_mcnp[v] / r_mcnp_binding
    #
    #   Tanto spnd_value como FLUX_CORR_FACTOR CANCELAN algebraicamente.
    #   → eps_S NO entra en las varianzas de dosis (hacerlo sería doble conteo).
    #   → eps_time tampoco (el tiempo no es una medición independiente).
    #   → Para el reporte de σ(t): σ(t)/t ≈ eps_S (SPND domina la incertidumbre
    #     del tiempo, pero esa incertidumbre no se transmite a las dosis).
    #
    # MODO TIEMPO FIJO:
    #   D[v] = r_mcnp[v] × (spnd_value/FLUX_CORR_FACTOR) × t_medido
    #   Todas las fuentes son independientes → eps_S y eps_time entran normalmente.

    _time_from_constr = mode_constraints or time_from_constraints

    if _time_from_constr:
        eps_S_dose    = 0.0   # spnd_value cancela en la ratio → no propaga a dosis
        eps_time_dose = 0.0   # tiempo derivado de constraints, no medición independiente
        # Incertidumbre del tiempo para reporte (el tiempo sí tiene sigma):
        # σ(t)/t ≈ eps_S  (eps_mcnp_binding se aproxima como despreciable)
        time_err_report = time_used * eps_S
    else:
        eps_S_dose    = eps_S
        eps_time_dose = eps_time
        time_err_report = abs(float(time_err)) if float(time_err) > 0 else 0.0

    # eps_rel para IsoE: incluye solo las fuentes que NO cancelaron.
    # En constraints: solo eps_sys (eps_S y eps_time cancelan en D[v]).
    # En tiempo fijo: eps_S, eps_time y eps_sys contribuyen en cuadratura.
    if _time_from_constr:
        eps_rel_for_isoe = eps_sys
    else:
        eps_rel_for_isoe = float(np.sqrt(eps_S**2 + eps_time**2 + eps_sys**2))

    # ════════════════════════════════════════════════════════════════════════
    # PASO 2 — Cálculo de dosis por vóxel
    # ════════════════════════════════════════════════════════════════════════

    Report = {
        "meta": {
            "date":             datetime.now(timezone.utc).isoformat(),
            "spnd":             float(spnd_value),
            "flux_corr_factor": float(FLUX_CORR_FACTOR),
            "Corr":             float(Corr),
            "time":             float(time_used),
            # sigma(t) calculado correctamente:
            #   modo constraints → time_used × eps_S
            #   modo tiempo fijo → incertidumbre absoluta del usuario
            "time_err":         float(time_err_report),
            # eps_S real (para referencia en reporte y trazabilidad)
            "eps_S":            float(eps_S),
            # eps_S efectivo usado en varianzas de dosis (0.0 en modo constraints)
            "eps_S_dose":       float(eps_S_dose),
            # eps_time efectivo usado en varianzas de dosis (0.0 en modo constraints)
            "eps_time":         float(eps_time_dose),
            "eps_sys":          float(eps_sys),
            # eps_rel pre-calculado para IsoE (excluye eps_S en modo constraints)
            "eps_rel_for_isoe": float(eps_rel_for_isoe),
            "mode":             "constraints" if mode_constraints else "time",
            "dose_for_limits":  dose_for_limits,
            "constraints_used": summarize_constraints_matrix(
                constraints_matrix, organ_order
            ) if (mode_constraints and constraints_matrix is not None) else [],
        },
        "ParamsUsed":      {},
        # Componentes de dosis física por vóxel × órgano (Boro, Fstn, Thn, Gamma)
        # + errores absolutos por componente (sigma_Boro, sigma_Fstn, ...)
        "CompDoses":       {},
        # Dosis física total por vóxel (suma lineal de componentes)
        "PhysVoxel":       {},
        # Dosis equivalente (pesada por RBE/CBE) por vóxel
        "BioVoxel":        {},
        # Error absoluto de dosis física por vóxel (propagación cuadrática correcta)
        "SigmaPhysVoxel":  {},
        # Error absoluto de dosis biológica por vóxel
        "SigmaBioVoxel":   {},
    }

    Dsum_phys: dict = {}
    Dsum_bio:  dict = {}
    Deq:       dict = {}

    # Acumulador para eps_rel_global al final
    _all_eps_phys: list[float] = []
    _all_eps_bio:  list[float] = []

    for i, organ_logic in enumerate(organ_order):
        # Resolver las claves reales del dict vectors para este órgano lógico
        keys_real: list[str] = []
        if organ_logic == "Pulmon":
            # El pulmón puede estar separado en izquierdo, derecho y/o total
            for k in ("PulmonIzq", "PulmonDer", "PulmonTotal"):
                if k in vectors:
                    keys_real.append(k)
        else:
            if organ_logic in vectors:
                keys_real.append(organ_logic)

        for key in keys_real:
            Br, Fr, Tr, Gr = vectors[key]
            Bval = float(B_arr[i])
            Berr = float(B_err_arr[i])
            CBE  = float(CBE_arr[i])
            RBE  = float(RBE_arr[i])

            # ── Dosis física por vóxel y componente [Gy] ─────────────────────
            D_b    = Br * Bval * Corr * time_used   # componente boro
            D_f    = Fr * Corr * time_used            # componente neutrones rápidos
            D_t    = Tr * Corr * time_used            # componente neutrones térmicos
            D_g    = Gr * Corr * time_used            # componente gamma
            D_phys = D_b + D_f + D_t + D_g           # dosis física total

            # ── Dosis equivalente H [Gy(RBE)] pesada por factores biológicos ─
            H = CBE * D_b + RBE * D_f + RBE * D_t + 1.0 * D_g

            # ── Errores relativos MCNP por vóxel (arrays) ────────────────────
            # Si la clave no existe en vectors_err se asume error MCNP = 0
            _zero = np.zeros(Br.shape, dtype=np.float64)
            _err_tuple = vectors_err.get(key)
            if _err_tuple is not None and len(_err_tuple) == 4:
                Br_err, Fr_err, Tr_err, Gr_err = (
                    np.asarray(e, dtype=np.float64) for e in _err_tuple
                )
                # Garantizar shapes compatibles (puede venir escalar 0 por defecto)
                Br_err = np.broadcast_to(Br_err, Br.shape).copy()
                Fr_err = np.broadcast_to(Fr_err, Fr.shape).copy()
                Tr_err = np.broadcast_to(Tr_err, Tr.shape).copy()
                Gr_err = np.broadcast_to(Gr_err, Gr.shape).copy()
            else:
                Br_err = Fr_err = Tr_err = Gr_err = _zero

            # ── Incertidumbre relativa de boro ───────────────────────────────
            # eps_B = sigma_B / Bval. Si Bval = 0, boro no contribuye y eps_B = 0.
            eps_B = (Berr / Bval) if Bval > 0 else 0.0

            # ── Propagación de incertidumbres con correlaciones correctas ─────
            #
            # eps_S_dose, eps_time_dose y eps_sys actúan como FACTORES GLOBALES
            # que escalan todos los componentes a la vez (a través de Corr y t).
            # Por eso D_b, D_f, D_t, D_g están CORRELACIONADOS entre sí por
            # esas fuentes. La varianza correcta de su suma D_phys es:
            #
            #   σ²(D_phys) = Σ σ²_indep(D_i) + Cov_cross
            #   Cov_cross  = (eps_S_dose² + eps_time_dose² + eps_sys²) × D_phys²
            #
            # La fórmula ingenua (sin cross-terms) usaría Σ D_i² en vez de D_phys²
            # y subestimaría σ cuando varios componentes son comparables.
            #
            # eps_B y eps_mcnp[v] son independientes entre componentes → no hay
            # cross-term entre ellos.
            #
            # En modo constraints: eps_S_dose = eps_time_dose = 0.0
            # porque spnd_value cancela en D[v] = D_lim × r_mcnp[v] / r_mcnp_binding.

            eps_corr_sq = eps_S_dose**2 + eps_time_dose**2 + eps_sys**2

            # Varianzas de fuentes independientes por componente:
            var_Db_ind = D_b**2 * (eps_B**2 + Br_err**2)
            var_Df_ind = D_f**2 * Fr_err**2
            var_Dt_ind = D_t**2 * Tr_err**2
            var_Dg_ind = D_g**2 * Gr_err**2

            # Error absoluto de dosis física:
            # suma de independientes + término correlacionado con D_phys²
            sigma_phys = np.sqrt(
                var_Db_ind + var_Df_ind + var_Dt_ind + var_Dg_ind
                + eps_corr_sq * D_phys**2
            )

            # Sigmas individuales por componente (para CompDoses):
            # incluyen todas sus fuentes relevantes
            sigma_Db = np.sqrt(D_b**2 * (eps_B**2 + Br_err**2 + eps_corr_sq))
            sigma_Df = np.sqrt(D_f**2 * (Fr_err**2 + eps_corr_sq))
            sigma_Dt = np.sqrt(D_t**2 * (Tr_err**2 + eps_corr_sq))
            sigma_Dg = np.sqrt(D_g**2 * (Gr_err**2 + eps_corr_sq))

            # Error relativo de dosis física por vóxel (para métricas y reporte)
            with np.errstate(invalid="ignore", divide="ignore"):
                eps_rel_phys_vox = np.where(D_phys > 0, sigma_phys / D_phys, 0.0)

            # ── Varianza de la dosis biológica H = CBE×D_b + RBE×(D_f+D_t) + D_g ──
            # Misma estructura que D_phys: fuentes independientes por componente +
            # término correlacionado con H² (no con Σ (CBE²D_b²+...) como antes).
            var_H_ind = (
                CBE**2 * var_Db_ind
                + RBE**2 * (var_Df_ind + var_Dt_ind)
                + var_Dg_ind
            )
            sigma_H = np.sqrt(var_H_ind + eps_corr_sq * H**2)

            with np.errstate(invalid="ignore", divide="ignore"):
                eps_rel_H_vox = np.where(H > 0, sigma_H / H, 0.0)

            # ── Métricas escalares representativas (media del eps vóxel a vóxel) ──
            eps_rel_phys_mean = float(np.nanmean(eps_rel_phys_vox))
            eps_rel_H_mean    = float(np.nanmean(eps_rel_H_vox))

            _all_eps_phys.append(eps_rel_phys_mean)
            _all_eps_bio.append(eps_rel_H_mean)

            # ── Guardar componentes, totales y errores por vóxel ─────────────
            Report["CompDoses"][key] = {
                "Boro":        D_b.tolist(),
                "Fstn":        D_f.tolist(),
                "Thn":         D_t.tolist(),
                "Gamma":       D_g.tolist(),
                # Errores absolutos por componente por vóxel
                "sigma_Boro":  sigma_Db.tolist(),
                "sigma_Fstn":  sigma_Df.tolist(),
                "sigma_Thn":   sigma_Dt.tolist(),
                "sigma_Gamma": sigma_Dg.tolist(),
            }
            Report["PhysVoxel"][key]      = D_phys.tolist()
            Report["BioVoxel"][key]       = H.tolist()
            Report["SigmaPhysVoxel"][key] = sigma_phys.tolist()
            Report["SigmaBioVoxel"][key]  = sigma_H.tolist()

            # ── Métricas por órgano ───────────────────────────────────────────
            # metrics_with_uncertainty recibe el eps_rel medio representativo
            Dsum_phys[key] = metrics_with_uncertainty(D_phys, eps_rel_phys_mean)
            Dsum_bio[key]  = metrics_with_uncertainty(H,      eps_rel_H_mean)

            H_mean = float(np.nanmean(H)) if H.size > 0 else 0.0
            H_max  = float(np.nanmax(H))  if H.size > 0 else 0.0

            # sigma en el vóxel de dosis máxima biológica
            _idx_max = int(np.nanargmax(H)) if H.size > 0 else 0
            sigma_H_at_max = float(sigma_H[_idx_max]) if H.size > 0 else 0.0

            Deq[key] = {
                "H_mean":          H_mean,
                "H_max":           H_max,
                # Errores absolutos
                "sigma_H_mean":    float(np.nanmean(sigma_H)),
                "sigma_H_max":     sigma_H_at_max,
                # Errores relativos medios del órgano
                "eps_rel_phys":    eps_rel_phys_mean,
                "eps_rel_bio":     eps_rel_H_mean,
                # Parámetros usados
                "CBE": CBE, "RBE": RBE,
                "B_used": Bval, "B_err": Berr,
            }

    Report["ParamsUsed"] = {
        "B":     B_arr.tolist(),
        "B_err": B_err_arr.tolist(),
        "CBE":   CBE_arr.tolist(),
        "RBE":   RBE_arr.tolist(),
    }

    # eps_rel global: promedio de todos los órganos procesados
    # (corrección del bug de overwrite — antes solo quedaba el último)
    Report["meta"]["eps_rel_global_phys"] = (
        float(np.nanmean(_all_eps_phys)) if _all_eps_phys else 0.0
    )
    Report["meta"]["eps_rel_global_bio"] = (
        float(np.nanmean(_all_eps_bio)) if _all_eps_bio else 0.0
    )

    if _dmin_conflict_info is not None:
        Report["meta"]["dmin_conflict"] = _dmin_conflict_info

    # ════════════════════════════════════════════════════════════════════════
    # PASO 3 — Identificación del constraint más cercano a saturación
    # ════════════════════════════════════════════════════════════════════════

    if mode_constraints and constraints_matrix is not None:
        best_ratio = -np.inf
        final_lim  = None
        rows       = constraints_matrix.shape[0]

        # Evaluar con el mapa de dosis correspondiente al modo seleccionado
        vec_map = Report["BioVoxel"] if dose_for_limits == "bio" else Report["PhysVoxel"]

        for i, organ_logic in enumerate(organ_order):
            key_eval = "PulmonTotal" if organ_logic == "Pulmon" else organ_logic
            if key_eval not in vec_map:
                continue

            vec = np.array(vec_map[key_eval])

            Dmax_allowed  = float(constraints_matrix[0, i]) if rows > 0 else 0.0
            Dmean_allowed = float(constraints_matrix[1, i]) if rows > 1 else 0.0
            Dmin_allowed  = float(constraints_matrix[2, i]) if rows > 2 else 0.0
            Vx_pct        = float(constraints_matrix[3, i]) if rows > 3 else 0.0
            Dose_at_Vx    = float(constraints_matrix[4, i]) if rows > 4 else 0.0

            def _check(allowed, achieved, label):
                nonlocal best_ratio, final_lim
                if allowed > 0:
                    ratio = achieved / allowed
                    if ratio > best_ratio:
                        best_ratio = ratio
                        final_lim  = (key_eval, label, allowed, achieved, ratio, i)

            _check(Dmax_allowed,  float(np.nanmax(vec)),  "Dmax")
            _check(Dmean_allowed, float(np.nanmean(vec)), "Dmean")
            if Vx_pct > 0 and Dose_at_Vx > 0:
                _check(Dose_at_Vx, float(np.percentile(vec, 100.0 - Vx_pct)), f"D{Vx_pct}%")
            _check(Dmin_allowed, float(np.nanmin(vec)), "Dmin")

        if final_lim:
            Report["meta"]["chosen_constraint"] = {
                "org":            final_lim[0],
                "type":           final_lim[1],
                "limit_value":    final_lim[2],
                "achieved_value": final_lim[3],
                "ratio":          final_lim[4],
                "time_computed":  float(time_used),
            }

    return Report, Dsum_phys, Dsum_bio, Deq
