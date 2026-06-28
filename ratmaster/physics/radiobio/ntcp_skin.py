"""
physics/radiobio/ntcp_skin.py
==============================
NTCP de piel para BNCT — adaptado de González et al. (2009).

Referencia:
    S.J. González et al., "Tumor control and normal tissue complications in
    BNCT treatment of nodular melanoma: A search for predictive quantities",
    Applied Radiation and Isotopes 67 (2009) S153–S156.

Modelo original (DAH — dosis-área histograma)
----------------------------------------------
El paper usa el modelo de sub-volumen equivalente (Ec. 2.1):

    p(D, ν) = exp(-(N0/ν^k) · exp(-α·D))

donde:
    D : dosis en el área ν de piel [Gy-Eq]
    ν : fracción de ÁREA TOTAL irradiada a dosis D (0 < ν ≤ 1)
    N0, k, α : coeficientes ajustados a datos de tolerancia de piel (fotón,
               fracción única — Ellis 1968, Hopewell 1990)

Importante: ν es una fracción de área (no un conteo de vóxeles). Cuando toda
la piel recibe una dosis uniforme D, ν = 1 y:
    p(D, 1) = exp(-N0 · exp(-α·D))

Para una distribución no homogénea, el modelo de subvolumen equivalente
(González & Carando, 2008, Math. Med. Biol.) generaliza la Ec. 2.1 sumando
las contribuciones de cada bin de dosis del DAH, cada uno ponderado por su
propia fracción de área ν_j:

    NTCP = exp( -Σ_j  (N0/ν_j^k) · exp(-α·D_j) · ν_j )

Esta es la forma correcta para combinar bins (a diferencia de promediar
"vóxel por vóxel" sin ponderar por su peso de área/volumen, que NO es
equivalente y diverge cuando el número de vóxeles es grande).

Adaptación para DVH (RatMaster sin DAH explícito)
-------------------------------------------------
RatMaster entrega dosis isoefectiva por vóxel de piel (DVH discreto, peso de
volumen uniforme por vóxel). Se construye un histograma de dosis con NBINS
bins; cada bin j tiene:
    D_j  = centro del bin [Gy_eq]
    ν_j  = fracción de volumen total de piel en ese bin (Σ_j ν_j = 1)

y se aplica la fórmula de subvolumen equivalente arriba. Esto reproduce
exactamente p(D,1) cuando toda la piel está en un único bin (dosis
uniforme), y converge de forma estable sin depender artificialmente del
número de vóxeles segmentados.

⚠ Nota sobre una versión anterior de este módulo
--------------------------------------------------
Una versión previa sumaba exp(-α·Aj) vóxel por vóxel sin ponderar por su
fracción de volumen (equivalente a usar ν_j = 1 vóxel en vez de ν_j =
1/N_vox). Esto hacía que NTCP → 0 artificialmente con segmentaciones de
muchos vóxeles, sin relación con la dosis real. Esta versión corrige eso
agrupando por bins de dosis y ponderando cada bin por su fracción de
volumen real, consistente con la Ec. 2.1 del paper.

Parámetros por defecto
-----------------------
Ajustados a datos de tolerancia de piel para fracción única (fotón):
    N0 = 2.0   (coef. de escala de población celular)
    k  = 0.5   (exponente de área/volumen)
    α  = 0.24  Gy⁻¹  (radiosensibilidad de piel para fracción única)

Estos valores reproducen aproximadamente la transición entre eritema y
ulceración observada por el paper alrededor de 18 Gy en 100 cm² de piel.

Figuras de mérito adicionales (paper, Sección 2.3):
    - PEUD100: dosis uniforme equivalente en 100 cm² con misma NTCP
    - D100_mean: dosis media en el 100 cm² con mayor dosis
    - D_max: dosis máxima puntual en piel

Nota sobre las unidades
-----------------------
La dosis de entrada A_arr es la dosis isoefectiva [Gy_eq fotónico] que
calcula isoe.py. El modelo del paper trabaja en "Gy-Eq" de fracción única,
que es exactamente esta magnitud. No se necesita conversión adicional.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros por defecto
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N0_SKIN: float = 2.0      # coef. de escala [adim]
DEFAULT_K_SKIN:  float = 0.5      # exponente de área/volumen [adim]
DEFAULT_ALPHA_SKIN: float = 0.24  # [Gy⁻¹] — radiosensibilidad de piel

# Dosis umbral de referencia de la literatura (González 2009)
SKIN_REF_DOSES_GY = [15.0, 18.0, 20.0]   # [Gy-Eq]

# Top-fracción de vóxeles usada como proxy de "100 cm²" del paper
DEFAULT_TOP_FRACTION: float = 0.10

# Número de bins de dosis para construir el histograma equivalente al DAH
DEFAULT_N_BINS: int = 64

# Área de referencia clínica del paper (González et al. 2009): ν=1 equivale
# a 100 cm² de piel humana. N0, k, α fueron ajustados con esta referencia,
# así que cualquier ν usado en la Ec. 2.1 DEBE ser una fracción de ESTA
# área, nunca una fracción del volumen/área total de piel segmentada del
# rat (que no tiene relación física con la escala de ajuste del modelo).
DEFAULT_A_REF_CM2: float = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Función de complicación por sub-volumen (Ec. 2.1 del paper)
# ─────────────────────────────────────────────────────────────────────────────

def p_subvolume(D: float | np.ndarray,
                nu: float | np.ndarray,
                N0: float = DEFAULT_N0_SKIN,
                k: float = DEFAULT_K_SKIN,
                alpha: float = DEFAULT_ALPHA_SKIN) -> np.ndarray:
    """
    Probabilidad de complicación en un sub-volumen de fracción de área ν,
    irradiado uniformemente a dosis D (Ec. 2.1 de González et al. 2009).

    p(D, ν) = exp(-(N0/ν^k) · exp(-α·D))

    Args:
        D  : dosis [Gy-Eq].
        nu : fracción de ÁREA TOTAL (0 < ν ≤ 1). ν=1 → todo el órgano.
    """
    D_arr  = np.asarray(D, float)
    nu_arr = np.asarray(nu, float)
    nu_safe = np.where(nu_arr > 1e-12, nu_arr, 1e-12)
    exponent = (N0 / nu_safe ** k) * np.exp(-alpha * D_arr)
    return np.exp(-np.clip(exponent, 0.0, 700.0))


def _dose_histogram(A: np.ndarray, n_bins: int = DEFAULT_N_BINS):
    """
    Construye histograma diferencial (centros de bin, fracción de volumen)
    a partir del array de dosis por vóxel. Equivalente al DAH del paper
    cuando se asume volumen ≈ área (peso uniforme por vóxel).

    Returns:
        (D_centers, nu_fracs) — ambos arrays de tamaño ≤ n_bins,
        con nu_fracs.sum() == 1.0 (bins vacíos se excluyen).
    """
    A = np.asarray(A, float)
    if A.size == 0:
        return np.array([]), np.array([])

    d_min, d_max = float(np.min(A)), float(np.max(A))
    if d_max - d_min < 1e-9:
        # Dosis uniforme (o prácticamente uniforme): un solo bin exacto
        return np.array([d_max]), np.array([1.0])

    counts, edges = np.histogram(A, bins=n_bins, range=(d_min, d_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = counts > 0
    nu = counts[mask].astype(float)
    nu = nu / nu.sum()
    return centers[mask], nu


# ─────────────────────────────────────────────────────────────────────────────
# NTCP de piel — modelo de subvolumen equivalente, ponderado por bins
# ─────────────────────────────────────────────────────────────────────────────

def ntcp_skin(
    A_arr: np.ndarray,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """
    NTCP de piel para una distribución de dosis por vóxel (DVH discreto).

    Se construye un histograma de dosis (equivalente al DAH del paper) y
    se aplica el modelo de subvolumen equivalente:

        NTCP = exp( -Σ_j (N0/ν_j^k) · exp(-α·D_j) · ν_j )

    donde ν_j es la fracción de volumen (≈ área) de piel en el bin j.

    Esta formulación reproduce exactamente p(D,1) = exp(-N0·exp(-α·D))
    cuando la dosis es uniforme, y es estable sin depender artificialmente
    del número de vóxeles segmentados.

    Args:
        A_arr  : dosis isoefectiva por vóxel de piel [Gy_eq].
        N0     : coeficiente de escala de población celular.
        k      : exponente de área/volumen.
        alpha  : radiosensibilidad para fracción única [Gy⁻¹].
        n_bins : número de bins del histograma de dosis.

    Returns:
        NTCP ∈ [0, 1].
    """
    A = np.asarray(A_arr, float)
    A = np.clip(A, 0.0, None)
    if A.size == 0:
        return 0.0

    D_j, nu_j = _dose_histogram(A, n_bins)
    if D_j.size == 0:
        return 0.0

    nu_safe = np.maximum(nu_j, 1e-12)
    terms = (N0 / nu_safe ** k) * np.exp(-alpha * D_j) * nu_j
    exponent = float(np.sum(terms))
    return float(np.exp(-np.clip(exponent, 0.0, 700.0)))


def ntcp_skin_from_dvh(
    dose_bins: np.ndarray,
    vol_fracs: np.ndarray,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
) -> float:
    """
    NTCP de piel desde DVH diferencial ya construido externamente
    (centros de dosis, fracción de volumen por bin).

    NTCP = exp( -Σ_j (N0/ν_j^k) · exp(-α·D_j) · ν_j )

    Args:
        dose_bins : centros de bins de dosis [Gy_eq].
        vol_fracs : fracción diferencial de volumen por bin (Σ ≈ 1).

    Returns:
        NTCP ∈ [0, 1].
    """
    D  = np.asarray(dose_bins, float)
    dv = np.asarray(vol_fracs, float)
    dv = np.clip(dv, 0.0, None)
    tot = dv.sum()
    if tot <= 0:
        return 0.0

    nu = dv / tot
    nu_safe = np.maximum(nu, 1e-12)
    terms = (N0 / nu_safe ** k) * np.exp(-alpha * D) * nu
    exponent = float(np.sum(terms))
    return float(np.exp(-np.clip(exponent, 0.0, 700.0)))


# ─────────────────────────────────────────────────────────────────────────────
# NTCP de piel desde GEOMETRÍA DE CAMPO REAL (caso RatMaster: campo
# cilíndrico parcial, no la segmentación completa de piel del rat)
# ─────────────────────────────────────────────────────────────────────────────

def ntcp_skin_field_geometry(
    A_arr_full_skin: np.ndarray,
    A_skin_total_cm2: float,
    A_ref_cm2: float = DEFAULT_A_REF_CM2,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    """
    NTCP de piel reescalando el DVH completo de piel del rat a la
    referencia de área del paper (100 cm²), usando el área geométrica
    real de la piel segmentada (no su volumen ni su nro. de vóxeles).

    Por qué reescalar por ÁREA y no usar la fracción de vóxeles
    --------------------------------------------------------------
    N0, k, α (González et al. 2009) fueron calibrados con ν = fracción
    del área de referencia A_ref = 100 cm². Si en cambio se usa la
    fracción de volumen/vóxeles de la piel segmentada del rat como ν
    (lo que hacía la versión anterior de este módulo), se evalúa la
    Ec. 2.1 en una escala de área que no tiene relación con la escala
    de calibración, y el resultado deja de ser comparable a los datos
    de tolerancia del paper.

    La corrección correcta NO es "recortar" la piel a la región del
    campo geométrico y asumir dosis=0 fuera de él — eso introduce un
    artefacto (el modelo tiene un "NTCP basal" de ~13.5% incluso a
    D=0 en ν=1, porque está calibrado sobre cohortes clínicas reales
    sin escalones abruptos de dosis). En cambio, se conserva el DVH
    REAL completo de la piel segmentada (que ya decae suavemente con
    la distancia al campo, por dispersión/penumbra física) y se
    reescala el EJE de ν: en vez de que cada bin del DVH pese como
    fracción del volumen total de piel del rat, pesa como fracción
    del ÁREA DE REFERENCIA de 100 cm², usando el área real de la piel
    segmentada (A_skin_total_cm2) como puente de conversión:

        ν_j = (fracción de volumen del bin j) · (A_skin_total_cm2 / A_ref_cm2)

    Esto es exactamente correcto cuando A_skin_total_cm2 es el área
    real (cm²) de la piel segmentada — no su volumen en cm³. Si lo que
    tenés es el área geométrica de SOLO el campo de irradiación
    (A_field ≈ 20.9 cm², el corte cilíndrico) y no de toda la piel
    segmentada, usá ese valor como A_skin_total_cm2 pero pasando como
    A_arr_full_skin SOLO los vóxeles de piel dentro de ese campo (no
    toda la segmentación) — ver nota más abajo.

    Args:
        A_arr_full_skin   : dosis isoefectiva por vóxel [Gy_eq] de la
                             región de piel cuya área real (cm²) se
                             conoce y se pasa en A_skin_total_cm2.
                             Puede ser TODA la piel segmentada (con su
                             área real en cm²) o SOLO los vóxeles del
                             campo geométrico (con A_field en cm²) —
                             ambos son válidos, lo que importa es que
                             el área en cm² corresponda exactamente al
                             conjunto de vóxeles que se está pasando.
        A_skin_total_cm2  : área real [cm²] de la región representada
                             por A_arr_full_skin (no volumen, no nro.
                             de vóxeles).
        A_ref_cm2         : área de referencia de calibración (100 cm²,
                             González et al. 2009).
        N0, k, alpha      : parámetros del modelo (Ec. 2.1).
        n_bins            : bins del histograma de dosis.

    Returns dict:
        NTCP_skin    — NTCP reescalado a la referencia de 100 cm²
        nu_total     — fracción de A_ref ocupada por la región pasada
        D_mean_Gy, D_max_Gy — estadística de dosis de la región
        A_total_cm2, A_ref_cm2
    """
    A = np.asarray(A_arr_full_skin, float)
    A = np.clip(A, 0.0, None)
    if A.size == 0 or A_skin_total_cm2 <= 0.0:
        return {
            "NTCP_skin": 0.0, "nu_total": 0.0,
            "D_mean_Gy": 0.0, "D_max_Gy": 0.0,
            "A_total_cm2": float(A_skin_total_cm2), "A_ref_cm2": float(A_ref_cm2),
        }

    nu_total = float(A_skin_total_cm2) / float(A_ref_cm2)

    D_j, w_j = _dose_histogram(A, n_bins)   # w_j: fracción de volumen, suma 1
    nu_j = w_j * nu_total                     # fracción de A_ref por bin
    nu_j_safe = np.maximum(nu_j, 1e-12)
    exponent = float(np.sum((N0 / nu_j_safe ** k) * np.exp(-alpha * D_j) * nu_j))
    exponent = float(np.clip(exponent, 0.0, 700.0))
    ntcp = float(np.exp(-exponent))

    return {
        "NTCP_skin":   ntcp,
        "nu_total":    nu_total,
        "D_mean_Gy":   float(np.mean(A)),
        "D_max_Gy":    float(np.max(A)),
        "A_total_cm2": float(A_skin_total_cm2),
        "A_ref_cm2":   float(A_ref_cm2),
    }


def ntcp_skin_single_dose_field(
    D_field_Gy: float,
    A_field_cm2: float,
    A_ref_cm2: float = DEFAULT_A_REF_CM2,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
) -> dict:
    """
    NTCP de piel para el caso simple: una dosis única representativa
    (ej. D_max o el valor de Cohen-Kerrich) en un campo de área conocida,
    SIN intentar reconstruir qué pasa en el resto del área de referencia.

    Esta es la forma más simple y honesta de aplicar la Ec. 2.1 a tu
    caso (campo cilíndrico ≈ 20.9 cm², dosis ≈ 28.5 Gy[IsoE]):

        NTCP_campo = p(D_field, ν_campo) = exp(-(N0/ν_campo^k)·exp(-α·D_field))

    con ν_campo = A_field_cm2 / A_ref_cm2.

    A diferencia de ntcp_skin_field_geometry(), esta función NO intenta
    cerrar el balance con "el resto de los 100 cm² a dosis menor" — sólo
    informa la probabilidad de complicación asociada a ESE subvolumen
    irradiado a ESA dosis, que es la cantidad que el paper realmente usa
    como FOM comparable entre pacientes/campos (junto con D100_mean y
    PEUD100). Es la opción recomendada cuando no se dispone del DVH
    completo de piel con su área real en cm², sino solo de la dosis y
    el área del campo de interés clínico (como en tu protocolo).

    Args:
        D_field_Gy  : dosis isoefectiva representativa del campo [Gy_eq]
                      (ej. 28.5 Gy[IsoE], tu valor escalado Cohen-Kerrich).
        A_field_cm2 : área geométrica real del campo [cm²] (ej. ≈20.9 cm²
                      de la media superficie cilíndrica r=1.66cm, L=4cm).
        A_ref_cm2   : área de referencia de calibración (100 cm²).
        N0, k, alpha: parámetros del modelo.

    Returns dict:
        NTCP_field, nu_field, D_field_Gy, A_field_cm2, A_ref_cm2
    """
    if A_field_cm2 <= 0.0:
        return {
            "NTCP_field": 0.0, "nu_field": 0.0,
            "D_field_Gy": float(D_field_Gy),
            "A_field_cm2": float(A_field_cm2), "A_ref_cm2": float(A_ref_cm2),
        }

    nu_field = float(A_field_cm2) / float(A_ref_cm2)
    exponent = (N0 / nu_field ** k) * np.exp(-alpha * float(D_field_Gy))
    exponent = float(np.clip(exponent, 0.0, 700.0))
    ntcp_field = float(np.exp(-exponent))

    return {
        "NTCP_field":  ntcp_field,
        "nu_field":    nu_field,
        "D_field_Gy":  float(D_field_Gy),
        "A_field_cm2": float(A_field_cm2),
        "A_ref_cm2":   float(A_ref_cm2),
    }


def peud_skin(
    A_arr: np.ndarray,
    top_fraction: float = DEFAULT_TOP_FRACTION,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
    tol: float = 1e-9,
) -> float:
    """
    PEUD (Probability-Equivalent Uniform Dose) en el top_fraction de vóxeles.

    Es la dosis uniforme D* en el top_fraction% de la piel (proxy de
    100 cm²) que produce la MISMA NTCP que esa sub-región con su
    distribución real. Se resuelve con p(D*, 1) = p_real, ya que dentro
    del sub-volumen top_fraction ν=1 (es el 100% de ESA región).

    Args:
        A_arr        : dosis isoefectiva por vóxel de piel [Gy_eq].
        top_fraction : fracción superior de vóxeles a considerar (0..1).

    Returns:
        PEUD [Gy_eq].
    """
    A = np.asarray(A_arr, float)
    A = np.clip(A, 0.0, None)
    if A.size == 0:
        return 0.0

    n_top = max(1, int(np.ceil(top_fraction * A.size)))
    A_top = np.sort(A)[::-1][:n_top]

    # NTCP real de la sub-región top (ella misma es ν=1 dentro de su propio
    # subconjunto, por lo que usamos el histograma del top directamente)
    ntcp_target = ntcp_skin(A_top, N0, k, alpha)

    if ntcp_target <= 0.0:
        return 0.0
    if ntcp_target >= 1.0:
        return float(np.max(A_top))

    # Bisección: encontrar D* tal que p(D*, 1) = ntcp_target
    # p(D,1) = exp(-N0*exp(-alpha*D))  →  monótona creciente en D
    lo, hi = 0.0, float(np.max(A_top)) * 3.0 + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = p_subvolume(mid, 1.0, N0, k, alpha)
        if val < ntcp_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float((lo + hi) / 2.0)


def skin_dose_stats(
    A_arr: np.ndarray,
    top_fraction: float = DEFAULT_TOP_FRACTION,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
) -> dict:
    """
    Estadísticas de dosis y figuras de mérito para piel (González 2009).

    Returns dict:
        NTCP_skin      — probabilidad de complicación [0,1]
        D_max_Gy       — dosis máxima en piel [Gy_eq]
        D_mean_Gy      — dosis media global en piel [Gy_eq]
        D_top_mean_Gy  — dosis media en top_fraction% con mayor dosis [Gy_eq]
        PEUD_top_Gy    — PEUD en top_fraction% de vóxeles [Gy_eq]
        frac_above_15/18/20 — fracción de piel con dosis ≥ umbral
        N0, k, alpha   — parámetros usados
        top_fraction   — fracción usada para PEUD y D_top_mean
        N_voxels       — número de vóxeles
    """
    A = np.asarray(A_arr, float)
    A = np.clip(A, 0.0, None)
    N = A.size

    if N == 0:
        return {
            "NTCP_skin": 0.0, "D_max_Gy": 0.0, "D_mean_Gy": 0.0,
            "D_top_mean_Gy": 0.0, "PEUD_top_Gy": 0.0,
            "frac_above_15": 0.0, "frac_above_18": 0.0, "frac_above_20": 0.0,
            "N0": N0, "k": k, "alpha": alpha,
            "top_fraction": top_fraction, "N_voxels": 0,
        }

    n_top = max(1, int(np.ceil(top_fraction * N)))
    A_sorted_desc = np.sort(A)[::-1]
    A_top = A_sorted_desc[:n_top]

    ntcp = ntcp_skin(A, N0, k, alpha)
    peud = peud_skin(A, top_fraction, N0, k, alpha)

    return {
        "NTCP_skin":     ntcp,
        "D_max_Gy":      float(np.max(A)),
        "D_mean_Gy":     float(np.mean(A)),
        "D_top_mean_Gy": float(np.mean(A_top)),
        "PEUD_top_Gy":   peud,
        "frac_above_15": float(np.mean(A >= 15.0)),
        "frac_above_18": float(np.mean(A >= 18.0)),
        "frac_above_20": float(np.mean(A >= 20.0)),
        "N0":            float(N0),
        "k":             float(k),
        "alpha":         float(alpha),
        "top_fraction":  float(top_fraction),
        "N_voxels":      int(N),
    }


def ntcp_skin_dose_curve(
    dose_Gy: np.ndarray,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
) -> np.ndarray:
    """
    NTCP de piel como función de dosis UNIFORME [Gy_eq] — curva
    dosis-respuesta de referencia (toda la piel a la misma dosis, ν=1).

    NTCP(D) = p(D, 1) = exp(-N0 · exp(-α·D))

    Nota: a diferencia de versiones anteriores, esta curva ya NO depende
    de N_vox — es la curva intrínseca del modelo, válida para cualquier
    segmentación. El punto del plan real (con distribución no uniforme)
    se calcula por separado con ntcp_skin() y en general NO cae sobre
    esta curva, porque la curva asume D uniforme en toda la piel mientras
    que el plan real tiene una distribución de dosis (D_max ≠ D_mean).

    Returns:
        NTCP_arr ∈ [0, 1] para cada dosis en dose_Gy.
    """
    d = np.asarray(dose_Gy, float)
    return p_subvolume(d, 1.0, N0, k, alpha)


def mc_ntcp_skin_uncertainty(
    A_arr: np.ndarray,
    sigma_A: np.ndarray,
    N0: float = DEFAULT_N0_SKIN,
    k: float = DEFAULT_K_SKIN,
    alpha: float = DEFAULT_ALPHA_SKIN,
    N_samples: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Incertidumbre de NTCP_skin por Monte Carlo.

    Returns dict:
        NTCP_mean, NTCP_std, NTCP_p5, NTCP_p95, samples
    """
    rng = rng or np.random.default_rng()
    A   = np.asarray(A_arr, float)
    sA  = np.asarray(sigma_A, float)
    if A.size == 0:
        empty = np.array([])
        return {"NTCP_mean": 0.0, "NTCP_std": 0.0,
                "NTCP_p5": 0.0, "NTCP_p95": 0.0, "samples": empty}

    noise    = rng.standard_normal((N_samples, A.size))
    A_samp   = np.clip(A[None, :] + sA[None, :] * noise, 0.0, None)
    samples  = np.array([
        ntcp_skin(A_samp[i], N0, k, alpha)
        for i in range(N_samples)
    ])
    return {
        "NTCP_mean": float(np.mean(samples)),
        "NTCP_std":  float(np.std(samples, ddof=1)),
        "NTCP_p5":   float(np.percentile(samples, 5)),
        "NTCP_p95":  float(np.percentile(samples, 95)),
        "samples":   samples,
    }
