"""
physics/isoe.py
===============
Modelo de Dosis Isoefectiva a fotones (IsoE) para BNCT.
Modelo MLQ (Modified Linear-Quadratic) con sinergia completa entre componentes.

Referencias:
    [1] González SJ & Santa Cruz GA, Radiat. Res. 178 (2012), pp. 609-621.
    [2] Dattoli Viegas AM et al. (2025) — cerebro normal, cinética biexponencial.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLUJO DE USO RECOMENDADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. Obtener órganos del reporte:   organs = list(report["CompDoses"].keys())
    2. Auto-asignar presets:          assignment = auto_assign_presets(organs)
    3. Construir params por órgano:   pbo = build_params_by_organ(assignment)
    4. Calcular IsoE:                 isoe_report, metrics = compute_isoe_from_report(
                                          report, params_by_organ=pbo)

    Los pasos 2-3 se pueden hacer en la UI (IsoEOrganParamsDialog) para que el
    usuario revise y ajuste la asignación antes de calcular.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELO FÍSICO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    E_BNCT = Σ_i α_i D_i  +  Σ_{i≤j} G_ij(θ) √(β_i β_j) D_i D_j

    La dosis isoefectiva A satisface:
        α_R A + G_R β_R A² = E_BNCT
    G_R puede ser:
        "fixed"    → escalar fijo del experimento de referencia (legacy)
        "computed" → G_R = G(t_ref, cinética lowLET) — reproduce BNCTar

CORRECCIONES respecto a versiones antiguas:
    ✓ G_R: modo "fixed" (legacy) o "computed" (reproduce BNCTar exactamente)
    ✓ Dmean: bisección escalar sobre media real de IsoE(t), no promedio de tiempos
    ✓ G_ij cruzados: forma exacta de González & Santa Cruz (2012), Apéndice II
      (fracción de dosis D_i/(D_i+D_j)) para cinética biexponencial.
      Para cinética monoexponencial se usa pair_weights() con scheme="dose".
    ✓ Soporte para cinética biexponencial por órgano
    ✓ Auto-asignación de preset por tipo de tejido
    ✓ compute_isoe_from_report acepta params_by_organ
"""

from math import sqrt
import numpy as np

from ratmaster.physics.compute_bnct import compute_bnct
from ratmaster.physics.dose_utils import metrics_with_uncertainty


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

_T0_MONO_DEFAULT_S: float = 3600.0   # 1 h — González 2012, Apéndice I

# Biexponencial por defecto (Ang et al. 1992 / Dattoli Viegas 2025)
_BIEXP_T0F_S:      float = 2520.0    # 0.7 h — componente rápida
_BIEXP_T0S_S:      float = 13680.0   # 3.8 h — componente lenta
_BIEXP_PF_LOWLET:  float = 0.38      # fracción rápida γ / referencia
_BIEXP_PS_LOWLET:  float = 0.62
_BIEXP_PF_HIGHLET: float = 0.20      # fracción rápida Boro, Thn, Fstn
_BIEXP_PS_HIGHLET: float = 0.80

# Nombres para la UI
REPAIR_MODEL_LABELS: dict[str, str] = {
    "monoexp": "Único tiempo de reparación",
    "biexp":   "Múltiples tiempos de reparación",
}

# Categorías de tejido: clave → {label, description, organ_hints}
TISSUE_CATEGORIES: dict[str, dict] = {
    "tumor": {
        "label":       "Tejido tumoral",
        "description": "Volúmenes tumorales: GTV, CTV, PTV, lesiones.",
        "organ_hints": ["tumor", "gtv", "ctv", "ptv", "melanoma",
                        "glioma", "gliosarcoma", "metastasis"],
        "color":       "#6A1B9A",
    },
    "normal_brain": {
        "label":       "Cerebro normal",
        "description": "Parénquima cerebral, tronco encefálico, SNC.",
        "organ_hints": ["brain", "cerebro", "cerebralnormal", "brainstem",
                        "troncoencefalico", "wholebrain", "normalcerebro"],
        "color":       "#1565C0",
    },
    "spinal_cord": {
        "label":       "Médula espinal",
        "description": "Médula espinal (modelo tardío de SNC).",
        "organ_hints": ["spinalcord", "medulaespinal", "cordaespinal", "spinal","medula"],
        "color":       "#1565C0",
    },
    "skin": {
        "label":       "Piel",
        "description": "Tejido cutáneo superficial.",
        "organ_hints": ["skin", "piel", "dermis", "cutaneo"],
        "color":       "#D38D00",
    },
    "mucosa": {
        "label":       "Mucosa",
        "description": "Mucosa oral, nasal o de vías aéreas.",
        "organ_hints": ["mucosa", "mucous", "oralmucosa", "mucosaoral"],
        "color":       "#2E7D32",
    },
    "normal_lung": {
        "label":       "Pulmón normal",
        "description": "Parénquima pulmonar, pulmón total, pulmón ipsi/contralateral.",
        "organ_hints": [
            "pulmon", "lung", "pulmontotal", "pulmonizq", "pulmonder",
            "pulmonipsi", "pulmoncontra", "normalpulmon", "normallung",
        ],
        "color":       "#00838F",
    },
    "liver": {
        "label":       "Hígado",
        "description": "Parénquima hepático.",
        "organ_hints": [
            "higado", "liver", "hepatico", "hepatic",
        ],
        "color":       "#558B2F",
    },
    "kidney": {
        "label":       "Riñón",
        "description": "Riñones (ipsi y contralateral).",
        "organ_hints": [
            "rinon", "kidney", "renal", "rinones",
            "kidneyipsi", "kidneycontra",
        ],
        "color":       "#6D4C41",
    },
    "salivary_gland": {
        "label":       "Glándulas salivales",
        "description": "Parótidas, submandibulares, sublinguales.",
        "organ_hints": [
            "parotida", "parotid", "submandibular", "sublingual",
            "salivary", "glandulasal", "glandulasalivary",
        ],
        "color":       "#F52217",
    },
    "bone_marrow": {
        "label":       "Médula ósea",
        "description": "Médula ósea activa (roja).",
        "organ_hints": [
            "bonemarrow", "medulaosea", "marrow", "redmarrow",
            "medulaactiva", "activemarrow",
        ],
        "color":       "#4527A0",
    },
    "eye": {
        "label":       "Ojo / cristalino",
        "description": "Globo ocular y cristalino.",
        "organ_hints": [
            "eye", "lens", "cristalino", "ojoder", "ojoizq",
            "globoocular", "eyeright", "eyeleft",
        ],
        "color":       "#0277BD",
    },
    "unknown": {
        "label":       "Sin clasificar",
        "description": "Tejido no identificado automáticamente.",
        "organ_hints": [],
        "color":       "#546E7A",
    },
}

# Descripción física de cada parámetro numérico (símbolo, unidad, descripción)
PARAM_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "aR": ("α_R", "Gy⁻¹",
           "Componente LINEAL de la curva de supervivencia de la RADIACIÓN DE "
           "REFERENCIA (fotones). Daño celular por un único evento "
           "ionizante, irreparable entre fracciones."),
    "bR": ("β_R", "Gy⁻²",
           "Componente CUADRÁTICA de la referencia. Daño por interacción de dos "
           "sublesiones. La razón α_R/β_R [Gy] indica sensibilidad al "
           "fraccionamiento del tejido de referencia."),
    "GR": ("G_R", "—",
           "Factor de Lea-Catcheside de la IRRADIACIÓN FOTÓNICA DE REFERENCIA. "
           "Solo activo cuando GR_mode='fixed'. Usar 1.0 para referencia aguda. "
           "Cuando GR_mode='computed', este valor es ignorado: G_R se calcula "
           "automáticamente desde t_ref_s y la cinética lowLET del tejido."),
    "GR_mode": ("GR_mode", "—",
                "Modo de cálculo de G_R de referencia. "
                "'fixed': usa el valor de GR directamente (legacy, por defecto). "
                "'computed': calcula G_R = G(t_ref_s, cinética lowLET), "
                "reproduciendo el esquema de BNCTar/González 2012."),
    "t_ref_s": ("t_ref", "s",
                "Tiempo de irradiación fotónica de REFERENCIA [s]. "
                "Solo relevante cuando GR_mode='computed'. "
                "González 2012 usa 30 min = 1800 s. "
                "Poner 0.0 para reproducir referencia aguda (G_R → 1.0)."),
    "aB": ("α_Boro", "Gy⁻¹",
           "Componente lineal para DOSIS DE BORO (partículas α/⁷Li de la reacción "
           "¹⁰B(n,α)). Alta LET → α_B >> α_R. Derivado de ajuste a supervivencia "
           "con BPA + neutrones."),
    "bB": ("β_Boro", "Gy⁻²",
           "Componente cuadrática para Boro. Pequeña en alta LET."),
    "aFn": ("α_Fstn", "Gy⁻¹",
            "Componente lineal para NEUTRONES RÁPIDOS (protones de retroceso "
            "H(n,n)p). LET intermedia-alta. Approx. estándar: aFn = aTh."),
    "bFn": ("β_Fstn", "Gy⁻²",
            "Componente cuadrática para neutrones rápidos. Approx.: bFn = bTh."),
    "aTh": ("α_Thn", "Gy⁻¹",
            "Componente lineal para NEUTRONES TÉRMICOS (protones de ¹⁴N(n,p)¹⁴C). "
            "LET intermedia. Approx. estándar: aTh = aFn."),
    "bTh": ("β_Thn", "Gy⁻²",
            "Componente cuadrática para neutrones térmicos. Approx.: bTh = bFn."),
    "aG": ("α_Gamma", "Gy⁻¹",
           "Componente lineal para FOTONES GAMMA. Baja LET. "
           "Approx. estándar: aG = aR."),
    "bG": ("β_Gamma", "Gy⁻²",
           "Componente cuadrática para gamma. Approx.: bG = bR."),
}

DEFAULT_T0_MAP: dict[str, float] = {
    "Boro":  _T0_MONO_DEFAULT_S,
    "Fstn":  _T0_MONO_DEFAULT_S,
    "Thn":   _T0_MONO_DEFAULT_S,
    "Gamma": _T0_MONO_DEFAULT_S,
    "R":     _T0_MONO_DEFAULT_S,   # solo informativo; GR se toma de params.GR
}


# ══════════════════════════════════════════════════════════════════════════════
# IsoEParams — contenedor de parámetros radiobiológicos
# ══════════════════════════════════════════════════════════════════════════════

class IsoEParams:
    """
    Parámetros radiobiológicos del modelo IsoE (MLQ con sinergia).

    Referencia fotónica (R):
        aR, bR  — α/β del tejido para fotones de referencia [Gy⁻¹, Gy⁻²]
        GR      — Factor Lea-Catcheside de la REFERENCIA (escalar fijo)

    Componentes BNCT:
        aB, bB   — Boro
        aTh, bTh — Neutrones Térmicos
        aFn, bFn — Neutrones Rápidos
        aG, bG   — Gamma del haz

    Cinética de reparación:
        repair_model = "monoexp" → usa t0_map (un t0 por componente)
        repair_model = "biexp"   → usa t0f_s, t0s_s, pf/ps por LET
    """

    NUMERIC_KEYS: list[str] = [
        "aR", "bR", "GR",
        "aB", "bB",
        "aFn", "bFn",
        "aTh", "bTh",
        "aG",  "bG",
    ]
    BIEXP_KEYS: list[str] = [
        "t0f_s", "t0s_s",
        "pf_lowLET", "ps_lowLET",
        "pf_highLET", "ps_highLET",
    ]
    # [FIX] META_KEYS: claves que controlan el modo de cálculo de G_R y el tiempo
    # de referencia. Antes no estaban en ninguna lista y build_params_by_organ las
    # filtraba silenciosamente, dejando GR_mode="fixed" (default) aunque el preset
    # dijera "computed". Esto causaba ~470 s de diferencia respecto al script de
    # validación. Ahora se exponen aquí para que cualquier función que construya
    # IsoEParams desde un dict pueda usar IsoEParams.ALL_KEYS como whitelist completa.
    META_KEYS: list[str] = ["repair_model", "GR_mode", "t_ref_s"]
    # ALL_KEYS: unión de todas las claves válidas — usar en build_params_by_organ
    # y en cualquier serialización/deserialización de IsoEParams.
    ALL_KEYS: list[str] = NUMERIC_KEYS + BIEXP_KEYS + META_KEYS

    def __init__(
        self,
        aR: float, bR: float, GR: float,
        aB: float, bB: float,
        aTh: float, bTh: float,
        aFn: float, bFn: float,
        aG: float,  bG: float,
        Gmix: float = 1.0,
        repair_model: str = "monoexp",
        t0f_s: float = _BIEXP_T0F_S,
        t0s_s: float = _BIEXP_T0S_S,
        pf_lowLET: float  = _BIEXP_PF_LOWLET,
        ps_lowLET: float  = _BIEXP_PS_LOWLET,
        pf_highLET: float = _BIEXP_PF_HIGHLET,
        ps_highLET: float = _BIEXP_PS_HIGHLET,
        GR_mode: str = "fixed",
        t_ref_s: float = 1800.0,
    ):
        self.aR  = float(aR);  self.bR  = float(bR);  self.GR  = float(GR)
        self.aB  = float(aB);  self.bB  = float(bB)
        self.aTh = float(aTh); self.bTh = float(bTh)
        self.aFn = float(aFn); self.bFn = float(bFn)
        self.aG  = float(aG);  self.bG  = float(bG)
        self.Gmix = float(Gmix)
        self.repair_model = str(repair_model)
        self.t0f_s      = float(t0f_s)
        self.t0s_s      = float(t0s_s)
        self.pf_lowLET  = float(pf_lowLET)
        self.ps_lowLET  = float(ps_lowLET)
        self.pf_highLET = float(pf_highLET)
        self.ps_highLET = float(ps_highLET)
        # GR_mode: "fixed"    → usa self.GR directamente (legacy, por defecto)
        #          "computed"  → calcula G_R desde t_ref_s y la cinética lowLET
        self.GR_mode = str(GR_mode)
        # t_ref_s: tiempo de irradiación fotónica de referencia [s]
        # Solo relevante cuando GR_mode == "computed".
        # 0.0 → referencia aguda (G_R = 1.0). 1800.0 → 30 min (González 2012).
        self.t_ref_s = float(t_ref_s)

    def copy(self) -> "IsoEParams":
        return IsoEParams(**self.to_dict())

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.NUMERIC_KEYS}
        d["repair_model"] = self.repair_model
        d.update({k: getattr(self, k) for k in self.BIEXP_KEYS})
        d["GR_mode"] = self.GR_mode
        d["t_ref_s"] = self.t_ref_s
        return d

    def __repr__(self) -> str:
        return (f"IsoEParams(aR={self.aR}, bR={self.bR}, GR={self.GR}, "
                f"GR_mode={self.GR_mode!r}, t_ref_s={self.t_ref_s}, "
                f"aB={self.aB}, aTh={self.aTh}, aFn={self.aFn}, "
                f"repair_model={self.repair_model!r})")


# ══════════════════════════════════════════════════════════════════════════════
# PRESETS DE LA LITERATURA
# ══════════════════════════════════════════════════════════════════════════════
# Estructura de cada preset:
# {
#   "ref":                  str  — cita bibliográfica completa
#   "tissue":               str  — descripción del tejido del modelo
#   "tissue_type":          str  — clave de TISSUE_CATEGORIES
#   "valid_organs":         list — órganos para los que aplica (vacío = todos)
#   "model_system":         str  — sistema experimental
#   "endpoint":             str  — endpoint medido
#   "boron_compound":       str  — compuesto de boro usado
#   "approximation_level":  str  — "full" | "standard" | "manual"
#   "approx_notes":         str  — qué se aproxima y por qué
#   "comments":             str  — notas de uso
#   "params":               dict — kwargs para IsoEParams (None = preset vacío)
#   "t0_map":               dict — {componente: t0_s} para monoexp (None si biexp)
# }

ISOE_PARAM_PRESETS: dict[str, dict] = {

    "Manual": {
        "ref":                "Entrada manual",
        "tissue":             "Sin especificar",
        "tissue_type":        "unknown",
        "valid_organs":       [],
        "model_system":       "—",
        "endpoint":           "—",
        "boron_compound":     "—",
        "approximation_level": "manual",
        "approx_notes":       "Parámetros ingresados por el usuario. Sin validación.",
        "comments":           "",
        "params":             None,
        "t0_map":             None,
    },
    "Tumor_DHD": {
        "ref": (
            "BIANCA fit, Università di Pavia"
        ),
        "tissue":         "Tumor",
        "tissue_type":    "tumor",
        "valid_organs":   [
            "Tumor"
        ],
        "model_system": (
            "—"
        ),
        "endpoint": (
            "—"
        ),
        "boron_compound": "—",
        "approximation_level": "full",
        "approx_notes": (
            "aTh = aFn, bTh = bFn (LET similar entre n térmicos y rápidos). aR = aG, bR = bG (fotones de referencia ≈ gamma del haz). GR = 1.0 (referencia aguda)."
        
        ),
        "comments": (
            ""
          
        ),
        "params": {
            "aR":  0.0342, "bR": 0.0286, "GR": 0.95,
            "aG":  0.0342, "bG": 0.0286,
            "aFn": 0.3759, "bFn": 0,
            "aTh": 0.3759, "bTh": 0,
            "aB": 1.2178, "bB": 0,
            "repair_model": "biexp",
            "t0f_s": 3200,
            "t0s_s": 3200,
            "pf_lowLET": 0.5,
            "ps_lowLET": 0.5,
            "pf_highLET": 0.5,
            "ps_highLET": 0.5,
            # G_R calculado desde el protocolo de referencia fotónica (no parámetro fijo).
            # t_ref = 1800 s (30 min): tiempo representativo de la irradiación de referencia
            # usado en el modelo publicado. Verificar con la fuente de αR/βR del preset.
            "GR_mode": "computed", "t_ref_s": 3200.0,
        },
        "t0_map": None,
    },

    "GS9L_Gonzalez2012": {
        "ref": (
            "González SJ & Santa Cruz GA. The photon-isoeffective dose in BNCT. "
            "Radiat. Res. 178 (2012) pp. 609-621. Tabla II — 9L rat gliosarcoma, "
            "in vivo/in vitro, CON sinergia."
        ),
        "tissue":         "Tumor gliosarcoma (9L, rata)",
        "tissue_type":    "tumor",
        "valid_organs":   ["Tumor", "GTV", "CTV"],
        "model_system":   "9L rat gliosarcoma, in vivo/in vitro, BMRR 1.25 MW",
        "endpoint":       "Supervivencia clonogénica S = 0.01",
        "boron_compound": "BPA",
        "approximation_level": "standard",
        "approx_notes": (
            "• Único tiempo de reparación t₀ = 1 h para todos los componentes "
            "(González 2012, Apéndice I; error < 3% para θ = 10–30 min).\n"
            "• aTh = aFn, bTh = bFn (LET similar para n térmicos y rápidos).\n"
            "• aGamma = aR, bGamma = bR (fotones del haz ≈ referencia).\n"
            "• GR = 1.0 (referencia de irradiación aguda, rayos X)."
        ),
        "comments": (
            "Ajuste simultáneo de datos beam-only y n+BPA con factor "
            "Lea-Catcheside explícito por punto. Recomendado para GTV/CTV "
            "en tratamientos de glioma con BPA."
        ),
        "params": {
            "aR": 0.2008, "bR": 0.0078, "GR": 1.0,
            "aG": 0.2008, "bG": 0.0078,
            "aFn": 0.4972, "bFn": 0.0880,
            "aTh": 0.4972, "bTh": 0.0880,
            "aB": 0.9091, "bB": 0.0019,
            "repair_model": "monoexp",
        },
        "t0_map": {
            "Boro": 3600.0, "Fstn": 3600.0,
            "Thn":  3600.0, "Gamma": 3600.0, "R": 3600.0,
        },
    },

    "MelJ_Gonzalez2012": {
        "ref": (
            "González SJ & Santa Cruz GA. The photon-isoeffective dose in BNCT. "
            "Radiat. Res. 178 (2012) pp. 609-621. Tabla III — Mel-J (melanoma "
            "humano), in vitro, CON sinergia."
        ),
        "tissue":         "Melanoma cutáneo (línea Mel-J, humano)",
        "tissue_type":    "tumor",
        "valid_organs":   ["Tumor", "GTV", "CTV", "Melanoma"],
        "model_system":   "Línea celular Mel-J (melanoma metastásico humano), in vitro, RA-6",
        "endpoint":       "Supervivencia clonogénica",
        "boron_compound": "BPA",
        "approximation_level": "standard",
        "approx_notes": (
            "• Único tiempo de reparación t₀ = 1 h.\n"
            "• aTh = aFn, bTh = bFn.\n"
            "• aGamma = aR, bGamma = bR.\n"
            "• GR = 1.0 (referencia aguda)."
        ),
        "comments": (
            "Derivado del estudio clínico de melanoma de Argentina (reactor RA-6). "
            "Validado contra datos de control tumoral (TCPMLQ). "
            "Usar solo para melanoma cutáneo nodular tratado con BPA."
        ),
        "params": {
            "aR": 0.0482, "bR": 0.0333, "GR": 1.0,
            "aG": 0.0482, "bG": 0.0333,
            "aFn": 0.5775, "bFn": 0.0464,
            "aTh": 0.5775, "bTh": 0.0464,
            "aB": 0.8156, "bB": 0.1021,
            "repair_model": "monoexp",
        },
        "t0_map": {
            "Boro": 3600.0, "Fstn": 3600.0,
            "Thn":  3600.0, "Gamma": 3600.0, "R": 3600.0,
        },
    },

    "CerebroNormal_Dattoli2025": {
        "ref": (
            "Dattoli Viegas AM, Carando D, Koivunoro H, Joensuu H, González SJ. "
            "Predicting radiotoxic effects after BNCT for brain cancer using a "
            "novel dose calculation model (2025). Tabla 2 — Médula espinal de rata, "
            "CON sinergia, cinética biexponencial."
        ),
        "tissue":         "Cerebro normal / Médula espinal (rata, SNC tardío)",
        "tissue_type":    "normal_brain",
        "valid_organs":   [
            "CerebralNormal", "BrainStem", "SpinalCord", "Brain",
            "Cerebro", "MedulaEspinal", "TroncoEncefalico", "WhileBrain",
        ],
        "model_system": (
            "Rata, médula espinal: fotones megavoltaje (Ang et al. 1987/1992, "
            "N=376 animales) y BNCT-BPA (Morris et al. 1994, BMRR, N=67 animales)."
        ),
        "endpoint": (
            "Parálisis de miembros (ED₅₀). NTCP Lyman: TD₅₀ = 23.04 Gy, m = 0.044. "
            "Efecto tardío (necrosis de sustancia blanca)."
        ),
        "boron_compound": "BPA intragástrico, 1500 mg/kg; ¹⁰B CNS = 10.0 ± 0.5 µg/g",
        "approximation_level": "full",
        "approx_notes": (
            "• Múltiples tiempos de reparación: t₀f = 0.7 h (rápida), "
            "t₀s = 3.8 h (lenta) — LET-independientes (Schmid et al. 2010).\n"
            "• Fracciones LET-específicas: low-LET (γ/ref) pf = 0.38, ps = 0.62; "
            "high-LET (B, Th, Fn) pf = 0.20, ps = 0.80 (Ang et al. 1992).\n"
            "• Parámetros normalizados: αR = 1.0 Gy⁻¹, α/β = 2 Gy → βR = 0.5 Gy⁻².\n"
            "• Ai = αi/αR, Bi = βi/αR según Tabla 2.\n"
            "• bFn = bTh ≈ 0 (neutrones actúan de forma lineal pura).\n"
            "• GR = 1.0 (referencia de megavoltaje, irradiación aguda)."
        ),
        "comments": (
            "Primer modelo DIsoE para cerebro normal con cinética biexponencial. "
            "Validado contra datos clínicos de somnolencia post-BNCT "
            "(Helsinki, FiR1 y BNL/Harvard-MIT). "
            "El DIsoE resultante está en Gy equivalente fotónico de megavoltaje. "
            "NO aplicar a volúmenes tumorales."
        ),
        "params": {
            "aR": 1.0, "bR": 0.5, "GR": 1.0,
            "aG": 1.0, "bG": 0.5,
            "aFn": 36.359, "bFn": 0.0,
            "aTh": 36.359, "bTh": 0.0,
            "aB": 21.7180, "bB": 0.8274,
            "repair_model": "biexp",
            "t0f_s": 2520, "t0s_s": 13680,
            "pf_lowLET": 0.38, "ps_lowLET": 0.62,
            "pf_highLET": 0.80, "ps_highLET": 0.20,
            # G_R calculado desde el protocolo de referencia fotónica (no parámetro fijo).
            # t_ref = 1800 s (30 min): tiempo representativo de la irradiación de referencia
            # usado en el modelo publicado. Verificar con la fuente de αR/βR del preset.
            "GR_mode": "computed", "t_ref_s": 1800.0,
        },
        "t0_map": None,
    },

    "Medula_Dattoli2025": {
        "ref": (
            "Dattoli Viegas AM, Carando D, Koivunoro H, Joensuu H, González SJ. "
            "Predicting radiotoxic effects after BNCT for brain cancer using a "
            "novel dose calculation model (2025). Tabla 2 — Médula espinal de rata, "
            "CON sinergia, cinética biexponencial."
        ),
        "tissue":         "Médula espinal (rata, SNC tardío)",
        "tissue_type":    "spinal_cord",
        "valid_organs":   [
            "Medula", "MedulaEspinal", "SpinalCord",
        ],
        "model_system": (
            "Rata, médula espinal: fotones megavoltaje (Ang et al. 1987/1992, "
            "N=376 animales) y BNCT-BPA (Morris et al. 1994, BMRR, N=67 animales)."
        ),
        "endpoint": (
            "Parálisis de miembros (ED₅₀). NTCP Lyman: TD₅₀ = 23.04 Gy, m = 0.044. "
            "Efecto tardío (necrosis de sustancia blanca)."
        ),
        "boron_compound": "BPA intragástrico, 1500 mg/kg; ¹⁰B CNS = 10.0 ± 0.5 µg/g",
        "approximation_level": "full",
        "approx_notes": (
            "• Mismos parámetros que el preset «CerebroNormal_Dattoli2025»: en el "
            "paper ambos tejidos (cerebro normal y médula espinal) se reportan con "
            "el mismo ajuste de Tabla 2, derivado de los datos de médula espinal de "
            "rata (Ang et al. 1987/1992; Morris et al. 1994) — no hay un set de "
            "parámetros separado para cerebro.\n"
            "• Múltiples tiempos de reparación: t₀f = 0.7 h (rápida), "
            "t₀s = 3.8 h (lenta) — LET-independientes (Schmid et al. 2010).\n"
            "• Fracciones LET-específicas: low-LET (γ/ref) pf = 0.38, ps = 0.62; "
            "high-LET (B, Th, Fn) pf = 0.20, ps = 0.80 (Ang et al. 1992).\n"
            "• Parámetros normalizados: αR = 1.0 Gy⁻¹, α/β = 2 Gy → βR = 0.5 Gy⁻².\n"
            "• Ai = αi/αR, Bi = βi/αR según Tabla 2.\n"
            "• bFn = bTh ≈ 0 (neutrones actúan de forma lineal pura).\n"
            "• GR = 1.0 (referencia de megavoltaje, irradiación aguda)."
        ),
        "comments": (
            "Mismo modelo que «CerebroNormal_Dattoli2025», separado como preset "
            "propio para que tissue_type coincida con la categoría «Médula "
            "espinal» y no dispare la advertencia de incompatibilidad de tejido "
            "en la UI. Validado contra datos clínicos de somnolencia post-BNCT "
            "(Helsinki, FiR1 y BNL/Harvard-MIT). El DIsoE resultante está en Gy "
            "equivalente fotónico de megavoltaje. NO aplicar a volúmenes tumorales."
        ),
        "params": {
            "aR": 1.0, "bR": 0.5, "GR": 1.0,
            "aG": 1.0, "bG": 0.5,
            "aFn": 36.359, "bFn": 0.0,
            "aTh": 36.359, "bTh": 0.0,
            "aB": 21.7180, "bB": 0.8274,
            "repair_model": "biexp",
            "t0f_s": 2520, "t0s_s": 13680,
            "pf_lowLET": 0.38, "ps_lowLET": 0.62,
            "pf_highLET": 0.80, "ps_highLET": 0.20,
            # G_R calculado desde el protocolo de referencia fotónica (no parámetro fijo).
            # t_ref = 1800 s (30 min): tiempo representativo de la irradiación de referencia
            # usado en el modelo publicado. Verificar con la fuente de αR/βR del preset.
            "GR_mode": "computed", "t_ref_s": 1800.0,
        },
        "t0_map": None,
    },

    "Piel_Dattoli2026": {
        "ref": (
            "An extended photon isoeffective dose model accounting"
            "for the energy-dependent effectiveness of secondary"
            "charged particles in BNCT (Dattoli Viegas et al. 2026). "
        ),
        "tissue":         "Piel",
        "tissue_type":    "skin",
        "valid_organs":   [
            "Piel", "piel", "Skin"
        ],
        "model_system": (
            "—"
        ),
        "endpoint": (
            "—"
        ),
        "boron_compound": "—",
        "approximation_level": "full",
        "approx_notes": (
            "aTh = aFn, bTh = bFn (LET similar entre n térmicos y rápidos). aR = aG, bR = bG (fotones de referencia ≈ gamma del haz). GR = 1.0 (referencia aguda)."
        
        ),
        "comments": (
            ""
          
        ),
        "params": {
            "aR": 0.1696, "bR": 0.0454, "GR": 1.0,
            "aG": 0.1696, "bG": 0.0454,
            "aFn": 1.5382, "bFn": 0.0516,
            "aTh": 1.5382, "bTh": 0.0516,
            "aB": 2.0028, "bB": 0.0037,
            "repair_model": "biexp",
            "t0f_s": 2164, "t0s_s": 6751.8,
            "pf_lowLET": 0.8, "ps_lowLET": 0.2,
            "pf_highLET": 0.35, "ps_highLET": 0.65,
            # G_R calculado desde el protocolo de referencia fotónica (no parámetro fijo).
            # t_ref = 1800 s (30 min): tiempo representativo de la irradiación de referencia
            # usado en el modelo publicado. Verificar con la fuente de αR/βR del preset.
            "GR_mode": "computed", "t_ref_s": 1800.0,
        },
        "t0_map": None,
    },
    "Pulmon_Normal": {
        "ref": (
            "BIANCA fit, Università di Pavia"
        ),
        "tissue":         "Pulmón normal",
        "tissue_type":    "normal_lung",
        "valid_organs":   [
            "Pulmon", "PulmonTotal", "PulmonIzq", "PulmonDer", "Lung","Pulmon Izquierdo", "Pulmon Derecho"
        ],
        "model_system": (
            "—"
        ),
        "endpoint": (
            "—"
        ),
        "boron_compound": "—",
        "approximation_level": "full",
        "approx_notes": (
            "aTh = aFn, bTh = bFn (LET similar entre n térmicos y rápidos). aR = aG, bR = bG (fotones de referencia ≈ gamma del haz). GR = 1.0 (referencia aguda)."
        
        ),
        "comments": (
            ""
          
        ),
        "params": {
            "aR":  0.112, "bR": 0.03, "GR": 1.0,
            "aG":  0.112, "bG": 0.03,
            "aFn": 0.7456, "bFn": 0.0183,
            "aTh": 0.7456, "bTh": 0.0183,
            "aB": 0.9242, "bB": 0.009,
            "repair_model": "biexp",
            "t0f_s": 1440, "t0s_s": 14436,
            "pf_lowLET": 0.8, "ps_lowLET": 0.2,
            "pf_highLET": 0.2, "ps_highLET": 0.8,
            # G_R calculado desde el protocolo de referencia fotónica (no parámetro fijo).
            # t_ref = 1800 s (30 min): tiempo representativo de la irradiación de referencia
            # usado en el modelo publicado. Verificar con la fuente de αR/βR del preset.
            "GR_mode": "computed", "t_ref_s": 1800.0,
        },
        "t0_map": None,
    },
    
}


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CLASIFICACIÓN Y AUTO-ASIGNACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def detect_tissue_type(organ_name: str) -> str:
    """
    Detecta el tipo de tejido de un órgano por su nombre.
    Retorna una clave de TISSUE_CATEGORIES ("tumor", "normal_brain", etc.).
    """
    nl = organ_name.lower().replace("_", "").replace(" ", "")
    for tissue_type, info in TISSUE_CATEGORIES.items():
        if tissue_type == "unknown":
            continue
        for hint in info["organ_hints"]:
            hint_clean = hint.lower().replace("_", "").replace(" ", "")
            if hint_clean in nl or nl in hint_clean:
                return tissue_type
    return "unknown"


def get_presets_for_tissue(tissue_type: str) -> list[str]:
    """
    Lista de presets compatibles con un tipo de tejido dado.
    Incluye presets con tissue_type = tissue_type.
    Excluye "Manual".
    """
    return [
        name for name, p in ISOE_PARAM_PRESETS.items()
        if name != "Manual" and p.get("tissue_type") == tissue_type
    ]


# ──────────────────────────────────────────────────────────────────────────────
# PREFERENCIAS DE AUTO-ASIGNACIÓN POR ÓRGANO
# ──────────────────────────────────────────────────────────────────────────────
# Cuando hay varios presets válidos para el mismo órgano (mismo tissue_type),
# get_presets_for_organ no tenía forma de saber cuál preferís como primera
# opción: el desempate caía en orden alfabético del nombre del preset. Por
# eso "Tumor" auto-asignaba GS9L_Gonzalez2012 en vez de Tumor_DHD (G < T).
#
# ORGAN_PRESET_PREFERENCE define, por órgano, el o los presets que querés
# que el auto-asignado elija primero. Es una lista (no un solo string) para
# poder tener más de un preset preferido por órgano: el primero de la lista
# que también sea válido para ese órgano gana; si ninguno aplica, se cae al
# desempate por defecto (tissue_type + alfabético).
#
# Para agregar preferencia en un órgano nuevo, alcanza con una entrada
# acá — no hace falta tocar get_presets_for_organ ni auto_assign_presets.
ORGAN_PRESET_PREFERENCE: dict[str, list[str]] = {
    "Tumor": ["Tumor_DHD"],
    "Medula": ["Medula_Dattoli2025"],
    # Ejemplo para extender a otro órgano con más de un preset preferido,
    # en orden de prioridad (el primero que aplique gana):
    # "Piel": ["Piel_Dattoli2026", "OtroPresetDePiel"],
}


def get_presets_for_organ(organ_name: str) -> list[str]:
    """
    Lista de presets cuyo valid_organs incluye organ_name,
    o cuyos valid_organs están vacíos (aplicable a cualquier órgano).
    Excluye "Manual". Ordenados por:
        1. Preferencia explícita en ORGAN_PRESET_PREFERENCE (si existe).
        2. Coincidencia de tissue_type.
        3. Nombre del preset (alfabético, desempate final).
    """
    detected_tt = detect_tissue_type(organ_name)
    preferred = ORGAN_PRESET_PREFERENCE.get(organ_name, [])
    matches: list[tuple[int, int, str]] = []
    for name, p in ISOE_PARAM_PRESETS.items():
        if name == "Manual":
            continue
        valid = p.get("valid_organs", [])
        if valid and organ_name not in valid:
            continue
        # Prioridad 1: posición en la lista de preferencia del órgano
        # (0 = primer preferido, 1 = segundo preferido, ..., N = sin preferencia).
        if name in preferred:
            pref_rank = preferred.index(name)
        else:
            pref_rank = len(preferred)
        # Prioridad 2: mismo tissue_type primero
        tt_rank = 0 if p.get("tissue_type") == detected_tt else 1
        matches.append((pref_rank, tt_rank, name))
    matches.sort()
    return [name for _, _, name in matches]


def auto_assign_presets(organ_list: list[str]) -> dict[str, str | None]:
    """
    Asigna automáticamente el mejor preset disponible para cada órgano.

    Para cada órgano:
        1. Detecta el tipo de tejido por su nombre.
        2. Busca presets con valid_organs que incluya el órgano y cuyo
           tissue_type coincida con el detectado.
        3. Si no hay coincidencia exacta, busca por tissue_type solo.
        4. Si no hay nada, asigna None (no se calculará IsoE).

    Retorna:
        {organ_name: preset_name | None}
    """
    result: dict[str, str | None] = {}
    for organ in organ_list:
        compatible = get_presets_for_organ(organ)
        result[organ] = compatible[0] if compatible else None
    return result


def build_params_by_organ(
    assignment: dict[str, str | None],
    user_presets: dict | None = None,
) -> dict[str, IsoEParams]:
    """
    Construye el diccionario {organ: IsoEParams} desde una asignación organ→preset.
    Órganos con preset=None son omitidos.

    Args:
        assignment:   resultado de auto_assign_presets() o de IsoEOrganParamsDialog.
        user_presets: USER_ISOE_PRESETS del módulo constants (puede ser None).

    Returns:
        {organ_key: IsoEParams} — solo para órganos con preset asignado.
    """
    all_presets = dict(ISOE_PARAM_PRESETS)
    if user_presets:
        all_presets.update(user_presets)

    result: dict[str, IsoEParams] = {}
    for organ, preset_name in assignment.items():
        if not preset_name:
            continue
        preset = all_presets.get(preset_name, {})
        raw_params = preset.get("params")
        if not raw_params:
            continue
        try:
            # [FIX] Antes: filtraba con NUMERIC_KEYS + BIEXP_KEYS + ["repair_model"],
            # lo que descartaba silenciosamente "GR_mode" y "t_ref_s".
            # Consecuencia: IsoEParams quedaba con GR_mode="fixed" (default del
            # constructor) aunque el preset dijera "computed", causando ~470 s de
            # diferencia en solve_time_isoe_from_constraints respecto al script de
            # validación.
            # Ahora: se usa ALL_KEYS que incluye META_KEYS = ["repair_model",
            # "GR_mode", "t_ref_s"], preservando el modo de cálculo correcto.
            p = IsoEParams(**{
                k: v for k, v in raw_params.items()
                if k in IsoEParams.ALL_KEYS
            })
            result[organ] = p
        except Exception:
            continue
    return result


def get_t0_map_for_preset(
    preset_name: str,
    user_presets: dict | None = None,
) -> dict:
    """Retorna el t0_map del preset, o DEFAULT_T0_MAP si no tiene."""
    all_presets = dict(ISOE_PARAM_PRESETS)
    if user_presets:
        all_presets.update(user_presets)
    p = all_presets.get(preset_name, {})
    t0 = p.get("t0_map")
    return dict(t0) if t0 else dict(DEFAULT_T0_MAP)


# ══════════════════════════════════════════════════════════════════════════════
# FACTORES DE LEA-CATCHESIDE
# ══════════════════════════════════════════════════════════════════════════════

def lea_catcheside(theta_s: float, t0_s: float) -> float:
    """G(θ, t₀) monoexponencial — escalar."""
    t = float(theta_s); t0 = max(float(t0_s), 1e-12)
    if t <= 1e-12:
        return 1.0
    x = t / t0
    return float((2.0 / x) * (1.0 + np.expm1(-x) / x))


def lea_catcheside_vec(theta: np.ndarray, t0_s: float) -> np.ndarray:
    """G(θ, t₀) monoexponencial — vectorizado."""
    theta = np.asarray(theta, float); t0 = max(float(t0_s), 1e-12)
    G = np.ones_like(theta)
    m = theta > 1e-12
    if m.any():
        x = theta[m] / t0
        G[m] = (2.0 / x) * (1.0 + np.expm1(-x) / x)
    return G


def lea_catcheside_biexp(theta_s: float, t0f: float, t0s: float,
                          pf: float, ps: float) -> float:
    """G_biexp = pf·G(θ,t0f) + ps·G(θ,t0s) — escalar."""
    return pf * lea_catcheside(theta_s, t0f) + ps * lea_catcheside(theta_s, t0s)


def lea_catcheside_biexp_vec(theta: np.ndarray, t0f: float, t0s: float,
                              pf: float, ps: float) -> np.ndarray:
    """G_biexp = pf·G(θ,t0f) + ps·G(θ,t0s) — vectorizado."""
    return pf * lea_catcheside_vec(theta, t0f) + ps * lea_catcheside_vec(theta, t0s)


def compute_GR(p: "IsoEParams", t0_map: dict) -> float:
    """
    Calcula G_R del experimento fotónico de referencia (escalar fijo).

    Reproduce el esquema de González 2012 / BNCTar:
        G_R = pf_lowLET·G(t_ref, t0f) + ps_lowLET·G(t_ref, t0s)   [biexp]
        G_R = G(t_ref, t0_R)                                         [monoexp]

    El tiempo de referencia t_ref (p.t_ref_s) corresponde al tiempo de
    irradiación fotónica del experimento de calibración (30 min en González 2012).
    Si p.t_ref_s es None o no existe, se usa 1800.0 s (30 min) como valor
    por defecto, que corresponde al tiempo representativo de la irradiación
    de referencia aguda de González 2012 (Apéndice I).

    Notas:
        - Para referencia AGUDA (dosis única instantánea), t_ref → 0 ⟹ G_R → 1.0.
          Fijar p.t_ref_s = 0.0 reproduce el comportamiento de GR = 1.0.
        - Este valor reemplaza al parámetro estático GR de IsoEParams cuando
          se activa el modo 'computed' (p.GR_mode == 'computed').
        - La cinética usada para G_R es siempre la de bajo LET (pf_lowLET,
          ps_lowLET), porque la referencia son fotones (bajo LET).
    """
    t_ref = getattr(p, "t_ref_s", None)
    if t_ref is None:
        t_ref = 1800.0   # 30 min — González 2012 Apéndice I

    t_ref = float(t_ref)

    if p.repair_model == "biexp":
        return lea_catcheside_biexp(t_ref, p.t0f_s, p.t0s_s, p.pf_lowLET, p.ps_lowLET)
    else:
        t0_R = t0_map.get("R", _T0_MONO_DEFAULT_S)
        return lea_catcheside(t_ref, t0_R)


def _resolve_GR(p: "IsoEParams", t0_map: dict) -> float:
    """
    Retorna el G_R efectivo a usar en la inversión IsoE.

    Si p.GR_mode == 'computed': calcula G_R desde t_ref y la cinética del tejido.
    Si p.GR_mode == 'fixed' (o ausente):  usa p.GR directamente (comportamiento legacy).
    """
    mode = getattr(p, "GR_mode", "fixed")
    if mode == "computed":
        return compute_GR(p, t0_map)
    return float(p.GR)


def _G_components(theta_s: float, p: IsoEParams, t0_map: dict) -> dict:
    """G_i(θ) por componente (escalar). NO incluye 'R' — GR = p.GR (fijo)."""
    if p.repair_model == "biexp":
        Gh = lea_catcheside_biexp(theta_s, p.t0f_s, p.t0s_s, p.pf_highLET, p.ps_highLET)
        Gl = lea_catcheside_biexp(theta_s, p.t0f_s, p.t0s_s, p.pf_lowLET,  p.ps_lowLET)
        return {"Boro": Gh, "Fstn": Gh, "Thn": Gh, "Gamma": Gl}
    return {k: lea_catcheside(theta_s, t0_map.get(k, _T0_MONO_DEFAULT_S))
            for k in ("Boro", "Fstn", "Thn", "Gamma")}


def _G_components_vec(theta: np.ndarray, p: IsoEParams, t0_map: dict) -> dict:
    """G_i(θ) por componente (vectorizado)."""
    if p.repair_model == "biexp":
        Gh = lea_catcheside_biexp_vec(theta, p.t0f_s, p.t0s_s, p.pf_highLET, p.ps_highLET)
        Gl = lea_catcheside_biexp_vec(theta, p.t0f_s, p.t0s_s, p.pf_lowLET,  p.ps_lowLET)
        return {"Boro": Gh, "Fstn": Gh, "Thn": Gh, "Gamma": Gl}
    return {k: lea_catcheside_vec(theta, t0_map.get(k, _T0_MONO_DEFAULT_S))
            for k in ("Boro", "Fstn", "Thn", "Gamma")}


def _Gij_cross(Di, Dj, pf_i: float, pf_j: float,
               Gf, Gs) -> np.ndarray:
    """
    Factor de Lea-Catcheside cruzado G_ij para un par de componentes BNCT.

    Para componentes i, j irradiados simultáneamente durante θ:

        a_ij = D_i / (D_i + D_j)    fracción de dosis del componente i
        a_ji = D_j / (D_i + D_j)

        G_ij = G_s - (a_ij · pf_i + a_ji · pf_j) · (G_s - G_f)

    donde:
        G_f = G(θ, τ_f)   extremo rápido de la cinética biexponencial
        G_s = G(θ, τ_s)   extremo lento
        pf_i               fracción de componente rápida del componente i
                           (pf_highLET para Boro/Thn/Fstn, pf_lowLET para γ)

    Cuando Di + Dj ≈ 0 en un vóxel, G_ij → G_s (la contribución al término
    cuadrático es nula de todas formas en ese vóxel).
    """
    Di_ = np.asarray(Di, float)
    Dj_ = np.asarray(Dj, float)
    eps = 1e-30
    s   = np.maximum(eps, Di_ + Dj_)
    a_ij = Di_ / s
    a_ji = Dj_ / s
    return Gs - (a_ij * pf_i + a_ji * pf_j) * (Gs - Gf)


# Se mantiene pair_weights como API de compatibilidad (monoexp / igual / dosis).
def pair_weights(Di, Dj, bi: float, bj: float,
                 scheme: str = "dose") -> tuple:
    """
    Pesos (a_i, a_j) para G_ij = a_i·G_i + a_j·G_j.

    NOTA: Para cinética biexponencial usar _Gij_cross() directamente,
    que reproduce la Ec. A5 de González & Santa Cruz (2012).

    "dose"  (González 2012, default): a_i = D_i / (D_i + D_j)
    "equal": a_i = a_j = 0.5
    "sublethal": a_i = sqrt(max(bi, 0.0)) * D_i / (sqrt(max(bi, 0.0)) * D_i + sqrt(max(bj, 0.0)) * D_j)
    """
    eps = 1e-50
    if scheme == "sublethal":
        if float(bi) == 0.0 and float(bj) == 0.0:
            return 0.5, 0.5
        ui = np.sqrt(max(bi, 0.0)) * np.asarray(Di, float)
        uj = np.sqrt(max(bj, 0.0)) * np.asarray(Dj, float)
        s  = np.maximum(eps, ui + uj)
        return ui / s, uj / s
    if scheme == "dose":
        Di_ = np.asarray(Di, float)
        Dj_ = np.asarray(Dj, float)
        s   = np.maximum(eps, Di_ + Dj_)
        return Di_ / s, Dj_ / s
    return 0.5, 0.5


# ══════════════════════════════════════════════════════════════════════════════
# FACTORIES DE CLOSURES — permiten params por órgano
# ══════════════════════════════════════════════════════════════════════════════

def _make_cmix_vec(p: IsoEParams, t0_map: dict, scheme: str):
    """
    Retorna Cmix_vec(Db, Df, Dt, Dg, theta) → ndarray
    Dosis = tasas × theta (irradiación a tasa constante).
    """
    aB, bB   = p.aB,  max(p.bB,  0.0)
    aFn, bFn = p.aFn, max(p.bFn, 0.0)
    aTh, bTh = p.aTh, max(p.bTh, 0.0)
    aG, bG   = p.aG,  max(p.bG,  0.0)

    def Cmix_vec(Db, Df, Dt, Dg, theta):
        theta = np.asarray(theta, float)
        dB, dF, dT, dG = Db*theta, Df*theta, Dt*theta, Dg*theta
        C = aB*dB + aFn*dF + aTh*dT + aG*dG
        Gi = _G_components_vec(theta, p, t0_map)
        C += (Gi["Boro"]*bB*dB*dB + Gi["Fstn"]*bFn*dF*dF +
              Gi["Thn"]*bTh*dT*dT + Gi["Gamma"]*bG*dG*dG)
        if p.repair_model == "biexp":
            # G_ij según González & Santa Cruz (2012), Apéndice II — fracción de dosis
            Gf = lea_catcheside_biexp_vec(theta, p.t0f_s, p.t0s_s, 1.0, 0.0)
            Gs = lea_catcheside_biexp_vec(theta, p.t0f_s, p.t0s_s, 0.0, 1.0)
            ph, pl = p.pf_highLET, p.pf_lowLET
            for Di, Dj, bi, bj, pf_i, pf_j in [
                (dB, dF, bB,  bFn, ph, ph),
                (dB, dT, bB,  bTh, ph, ph),
                (dB, dG, bB,  bG,  ph, pl),
                (dF, dT, bFn, bTh, ph, ph),
                (dF, dG, bFn, bG,  ph, pl),
                (dT, dG, bTh, bG,  ph, pl),
            ]:
                Gij = _Gij_cross(Di, Dj, pf_i, pf_j, Gf, Gs)
                C += 2.0 * Gij * sqrt(max(bi, 0.0)) * sqrt(max(bj, 0.0)) * Di * Dj
        else:
            for Di, Dj, bi, bj, ni, nj in [
                (dB, dF, bB,  bFn, "Boro",  "Fstn"),
                (dB, dT, bB,  bTh, "Boro",  "Thn"),
                (dB, dG, bB,  bG,  "Boro",  "Gamma"),
                (dF, dT, bFn, bTh, "Fstn",  "Thn"),
                (dF, dG, bFn, bG,  "Fstn",  "Gamma"),
                (dT, dG, bTh, bG,  "Thn",   "Gamma"),
            ]:
                ai, aj = pair_weights(Di, Dj, bi, bj, scheme)
                Gij = ai*Gi[ni] + aj*Gi[nj]
                C += 2.0 * Gij * sqrt(max(bi, 0.0)) * sqrt(max(bj, 0.0)) * Di * Dj
        return C

    return Cmix_vec


def _make_isoe_inv(p: IsoEParams, t0_map: dict | None = None):
    """
    Retorna isoe_inv(Cmix) → ndarray
    Resuelve α_R A + G_R β_R A² = Cmix.

    G_R se resuelve via _resolve_GR(p, t0_map):
        GR_mode="fixed"    → usa p.GR directamente (legacy)
        GR_mode="computed" → calcula G_R con la cinética lowLET
                             evaluada en p.t_ref_s (tiempo de referencia).
    """
    if t0_map is None:
        t0_map = dict(DEFAULT_T0_MAP)
    aR = p.aR
    bR = max(p.bR, 0.0)
    GR = _resolve_GR(p, t0_map)   # escalar — fijo o calculado

    def isoe_inv(Cmix):
        if GR * bR > 0.0:
            disc = np.maximum(0.0, aR*aR + 4.0*GR*bR*Cmix)
            return np.maximum(0.0, (-aR + np.sqrt(disc)) / (2.0*GR*bR))
        return np.maximum(0.0, Cmix / max(aR, 1e-30))

    return isoe_inv


# ══════════════════════════════════════════════════════════════════════════════
# compute_isoe_from_report
# ══════════════════════════════════════════════════════════════════════════════

def compute_isoe_from_report(
    report: dict,
    params: IsoEParams | None = None,
    synergy: bool = True,
    t0_map: dict | None = None,
    weight_scheme: str = "sublethal",
    params_by_organ: dict[str, IsoEParams] | None = None,
    auto_assign: bool = False,
    eps_rel: float | None = None,
) -> tuple[dict, dict]:
    """
    Calcula la dosis isoefectiva (Gy fotónico equivalente) por vóxel.

    IMPORTANTE — órganos sin parámetros:
        Si params_by_organ está definido y un órgano NO está en el dict,
        ese órgano se OMITE del cálculo (aparece en skipped_organs).
        Pasar params=None garantiza que no hay fallback para órganos no asignados.

    Incerteza (eps_rel):
        Si se provee eps_rel explícitamente, se usa ese valor.
        Si no, se intenta leer de report["meta"] (varios campos posibles).
        La propagación es: σ_A/A ≈ eps_rel*(C_lin + 2*C_quad)/C_total
        donde C_lin son los términos lineales y C_quad los cuadráticos del MLQ.
        Los términos cuadráticos tienen el doble de incerteza relativa que los
        lineales porque van como D² (ambos factores tienen incerteza eps_rel).
    """
    if t0_map is None:
        t0_map = dict(DEFAULT_T0_MAP)

    theta = float((report.get("meta") or {}).get("time", 1.0))
    theta = max(theta, 1e-12)

    # Resolver eps_rel: argumento explícito > meta del reporte > 0
    if eps_rel is not None:
        eps_rel_used = float(eps_rel)
    else:
        meta = report.get("meta") or {}
        eps_rel_used = float(
            meta.get("eps_rel_for_isoe")  # campo canónico calculado por compute_bnct
            or meta.get("eps_rel")
            or meta.get("combined_err")
            or meta.get("tot_err")
            or meta.get("rel_err")
            or 0.0
        )

    # Auto-asignación si se solicita
    if auto_assign and params_by_organ is None:
        organs = list(report.get("CompDoses", {}).keys())
        assignment = auto_assign_presets(organs)
        params_by_organ = build_params_by_organ(assignment)

    skipped:      list[str] = []
    unvalidated:  list[str] = []
    presets_used: dict[str, str] = {}
    IsoVoxel:     dict = {}
    SigmaIsoVoxel: dict = {}   # σ_A por vóxel — mismo esquema que SigmaPhysVoxel
    metrics:      dict = {}

    for key, comp in report.get("CompDoses", {}).items():
        # Selección de parámetros: params_by_organ tiene prioridad.
        # Si un órgano no está en params_by_organ Y params=None → se omite.
        p = (params_by_organ or {}).get(key)
        if p is None and params is not None:
            p = params   # fallback legacy (solo cuando params != None)
        if p is None:
            skipped.append(key)
            continue

        # Advertencia de validación
        preset_name = next(
            (n for n, pr in ISOE_PARAM_PRESETS.items()
             if pr.get("params") == p.to_dict()), None
        )
        if preset_name:
            valid_o = ISOE_PARAM_PRESETS[preset_name].get("valid_organs", [])
            if valid_o and key not in valid_o:
                unvalidated.append(key)
            presets_used[key] = preset_name

        Db = np.asarray(comp.get("Boro",  []), float)
        Df = np.asarray(comp.get("Fstn",  []), float)
        Dt = np.asarray(comp.get("Thn",   []), float)
        Dg = np.asarray(comp.get("Gamma", []), float)

        Gi = _G_components(theta, p, t0_map)
        aB, bB   = p.aB,  max(p.bB,  0.0)
        aFn, bFn = p.aFn, max(p.bFn, 0.0)
        aTh, bTh = p.aTh, max(p.bTh, 0.0)
        aG, bG   = p.aG,  max(p.bG,  0.0)

        # C_lin: términos lineales de E_mix
        C_lin = aB*Db + aFn*Df + aTh*Dt + aG*Dg

        # C_quad: términos cuadráticos (diagonal + cruzados)
        C_quad = (Gi["Boro"]*bB*Db*Db + Gi["Fstn"]*bFn*Df*Df +
                  Gi["Thn"]*bTh*Dt*Dt + Gi["Gamma"]*bG*Dg*Dg)
        if p.repair_model == "biexp":
            # G_ij según González & Santa Cruz (2012), Apéndice II — fracción de dosis
            Gf_s = lea_catcheside_biexp(theta, p.t0f_s, p.t0s_s, 1.0, 0.0)
            Gs_s = lea_catcheside_biexp(theta, p.t0f_s, p.t0s_s, 0.0, 1.0)
            ph, pl = p.pf_highLET, p.pf_lowLET
            for Di, Dj, bi, bj, pf_i, pf_j in [
                (Db, Df, bB,  bFn, ph, ph),
                (Db, Dt, bB,  bTh, ph, ph),
                (Db, Dg, bB,  bG,  ph, pl),
                (Df, Dt, bFn, bTh, ph, ph),
                (Df, Dg, bFn, bG,  ph, pl),
                (Dt, Dg, bTh, bG,  ph, pl),
            ]:
                Gij = _Gij_cross(Di, Dj, pf_i, pf_j, Gf_s, Gs_s)
                C_quad += 2.0 * Gij * sqrt(max(bi, 0.0)) * sqrt(max(bj, 0.0)) * Di * Dj
        else:
            for Di, Dj, bi, bj, ni, nj in [
                (Db, Df, bB,  bFn, "Boro",  "Fstn"),
                (Db, Dt, bB,  bTh, "Boro",  "Thn"),
                (Db, Dg, bB,  bG,  "Boro",  "Gamma"),
                (Df, Dt, bFn, bTh, "Fstn",  "Thn"),
                (Df, Dg, bFn, bG,  "Fstn",  "Gamma"),
                (Dt, Dg, bTh, bG,  "Thn",   "Gamma"),
            ]:
                ai, aj = pair_weights(Di, Dj, bi, bj, weight_scheme)
                Gij = ai*Gi[ni] + aj*Gi[nj]
                C_quad += 2.0 * Gij * sqrt(max(bi, 0.0)) * sqrt(max(bj, 0.0)) * Di * Dj

        C = C_lin + C_quad
        A = _make_isoe_inv(p, t0_map)(C)

        # Propagación de incerteza:
        # σ_A/A ≈ eps_rel_A donde:
        #   términos lineales (∝ D):  σ/valor = eps_rel
        #   términos cuadráticos (∝ D²): σ/valor = 2*eps_rel
        # => eps_rel_A = eps_rel * (C_lin + 2*C_quad) / max(C, ε)
        safe_C = np.maximum(np.abs(C), 1e-30)
        eps_rel_A = eps_rel_used * (C_lin + 2.0 * C_quad) / safe_C

        # eps_rel_A es array por vóxel; metrics_with_uncertainty necesita un escalar.
        # Usamos la media como valor representativo del órgano.
        eps_rel_scalar = float(np.mean(eps_rel_A)) if eps_rel_A.size > 0 else eps_rel_used

        # σ_A por vóxel: necesario para propagación en TCP/NTCP radiobiológico.
        # Se almacena en SigmaIsoVoxel con la misma estructura que SigmaPhysVoxel.
        sigma_A = A * np.maximum(eps_rel_A, 0.0)

        IsoVoxel[key]      = A.tolist()
        SigmaIsoVoxel[key] = sigma_A.tolist()
        metrics[key]       = metrics_with_uncertainty(A, eps_rel_scalar)

    isoe_report = {
        "IsoVoxel":      IsoVoxel,
        "SigmaIsoVoxel": SigmaIsoVoxel,   # σ(A) por vóxel — para propagación radiobiológica
        "meta": {
            "theta_s":            float(theta),
            "eps_rel_used":       float(eps_rel_used),
            "weight_scheme":      weight_scheme,
            "model":              "MLQ sinergia completa (González 2012 / Dattoli Viegas 2025)",
            "skipped_organs":     skipped,
            "unvalidated_organs": unvalidated,
            "organ_presets_used": presets_used,
            "per_organ_params":   params_by_organ is not None,
            "GR_note": (
                "G_R: modo usado por órgano. "
                "'fixed': valor de p.GR ingresado por el usuario. "
                "'computed': calculado como G(t_ref_s, cinética lowLET) "
                "del tejido — reproduce BNCTar/González 2012. "
                "En ningún caso depende del tiempo de irradiación BNCT θ."
            ),
        },
    }
    return isoe_report, metrics


# ══════════════════════════════════════════════════════════════════════════════
# solve_time_isoe_from_constraints
# ══════════════════════════════════════════════════════════════════════════════

def solve_time_isoe_from_constraints(
    vectors: dict,
    organ_order: list,
    B_arr,
    spnd_value: float,
    sys_error: float,
    spnd_error: float,
    params: IsoEParams,
    synergy: bool,
    constraints_matrix,
    t0_map: dict | None = None,
    weight_scheme: str = "sublethal",
    params_by_organ: dict[str, IsoEParams] | None = None,
) -> tuple[float, dict, None]:
    """
    Tiempo de irradiación óptimo para cumplir constraints IsoE.

    Correcciones:
        ✓ Dmean: bisección escalar sobre mean(IsoE(t)) real.
        ✓ Bracketing inicial estimado linealmente (sin duplicaciones ciegas).
        ✓ Convergencia por tolerancia en dosis [Gy], no en tiempo.
        ✓ GR = p.GR escalar fijo (no función de θ_BNCT).
        ✓ Soporte params_by_organ.
    """
    if t0_map is None:
        t0_map = dict(DEFAULT_T0_MAP)

    # Tasas de dosis a t=1 s
    report1, _, _, _ = compute_bnct(
        vectors=vectors,
        organ_order=organ_order,
        B_arr=np.array(B_arr, float),
        B_err_arr=np.zeros(len(organ_order), float),
        CBE_arr=np.ones(len(organ_order), float),
        RBE_arr=np.ones(len(organ_order), float),
        spnd_value=spnd_value, time_s=1.0, time_err=0.0,
        mode_constraints=False, constraints_matrix=None,
        sys_error=sys_error, spnd_error=spnd_error, dose_for_limits="phys",
    )
    comp = report1.get("CompDoses", {})
    if not comp:
        raise RuntimeError("No hay CompDoses para calcular IsoE (t=1s).")

    rates = {
        key: (
            np.asarray(cd.get("Boro",  []), float),
            np.asarray(cd.get("Fstn",  []), float),
            np.asarray(cd.get("Thn",   []), float),
            np.asarray(cd.get("Gamma", []), float),
        )
        for key, cd in comp.items()
    }

    upper_times: list[float] = []
    upper_tags:  list[tuple] = []
    dmin_times:  list[float] = []
    dmin_tags:   list[tuple] = []

    for i, organ_logic in enumerate(organ_order):
        key = "PulmonTotal" if organ_logic == "Pulmon" else organ_logic
        if key not in rates:
            continue

        p = (params_by_organ or {}).get(key) or params
        if p is None:
            continue

        Db, Df, Dt, Dg = rates[key]
        if Db.size == 0:
            continue

        cv   = _make_cmix_vec(p, t0_map, weight_scheme)
        inv  = _make_isoe_inv(p, t0_map)
        aR_  = p.aR
        bR_  = max(p.bR, 0.0)
        #cambio para incluir calculo con g computed en vez de fijo
        GR_  = _resolve_GR(p, t0_map)

        def _C_R(A): return aR_*A + GR_*bR_*A*A

        def _roots(A_target, tol=0.02, maxiter=60,
                   _Db=Db, _Df=Df, _Dt=Dt, _Dg=Dg):
            N = _Db.size
            C_R_t = _C_R(A_target)
            alpha_sum = np.maximum(
                p.aB*_Db + p.aFn*_Df + p.aTh*_Dt + p.aG*_Dg, 1e-30
            )
            t_hi = np.maximum(C_R_t / alpha_sum * 4.0, 5.0)
            t_lo = np.zeros(N)
            for _ in range(20):
                mask = cv(_Db, _Df, _Dt, _Dg, t_hi) < C_R_t
                if not mask.any(): break
                t_hi[mask] *= 2.0
            for _ in range(maxiter):
                tm = 0.5*(t_lo+t_hi)
                err = inv(cv(_Db, _Df, _Dt, _Dg, tm)) - A_target
                t_hi[err >= 0] = tm[err >= 0]
                t_lo[err <  0] = tm[err <  0]
                if np.abs(err).max() <= tol: break
            return t_hi

        def _mean_bisect(A_mean, tol=0.02, maxiter=80,
                         _Db=Db, _Df=Df, _Dt=Dt, _Dg=Dg):
            ams = float(np.nanmean(p.aB*_Db+p.aFn*_Df+p.aTh*_Dt+p.aG*_Dg))
            ams = max(ams, 1e-30)
            t_lo, t_hi = 0.0, max(_C_R(A_mean)/ams*4.0, 300.0)

            def mi(t):
                return float(np.nanmean(
                    inv(cv(_Db, _Df, _Dt, _Dg,
                           np.full(_Db.shape, float(t), float)))
                ))

            for _ in range(30):
                if mi(t_hi) >= A_mean: break
                t_hi *= 2.0
            if mi(t_lo) >= A_mean: return t_lo
            for _ in range(maxiter):
                tm = 0.5*(t_lo+t_hi)
                fm = mi(tm) - A_mean
                if abs(fm) <= tol: return tm
                if fm < 0: t_lo = tm
                else:       t_hi = tm
                if t_hi-t_lo < 0.5: break
            return 0.5*(t_lo+t_hi)

        def _c(row):
            return float(constraints_matrix[row, i]) if i < constraints_matrix.shape[1] else 0.0

        Dmax = _c(0); Dmean = _c(1); Dmin = _c(2); Vx = _c(3); dVx = _c(4)

        if Dmax > 0:
            upper_times.append(float(np.nanmin(_roots(Dmax))))
            upper_tags.append((key, "Dmax", Dmax, i))

        if Dmean > 0:
            upper_times.append(_mean_bisect(Dmean))
            upper_tags.append((key, "Dmean", Dmean, i))

        if Dmin > 0:
            dmin_times.append(float(np.nanmax(_roots(Dmin))))
            dmin_tags.append((key, "Dmin", Dmin, i))

        if 0 < Vx <= 100 and dVx > 0:
            tv = _roots(dVx)
            tv = tv[np.isfinite(tv) & (tv >= 0)]
            if tv.size > 0:
                srt = np.sort(tv)
                idx = max(0, min(srt.size-1, int(np.ceil(Vx*0.01*srt.size))-1))
                upper_times.append(float(srt[idx]))
                upper_tags.append((key, f"D{Vx}%", dVx, i))

    if not upper_times and not dmin_times:
        raise RuntimeError("No se generaron candidatos IsoE desde constraints.")

    all_t = upper_times + dmin_times
    all_g = upper_tags  + dmin_tags
    j = int(np.nanargmin(all_t))
    organ_key, metric, lv, iorg = all_g[j]

    return float(all_t[j]), {
        "org": organ_key, "type": metric,
        "limit_value": lv, "organ_index": int(iorg),
        "time_computed": float(all_t[j]),
    }, None
