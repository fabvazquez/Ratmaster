"""
constants.py
============
Constantes globales de RatMaster: orden de órganos, factores de calibración,
paleta de colores para DVH, y presets de protocolos de boro, RBE, CBE y constraints.

Estas definiciones son compartidas por los módulos de física, datos y UI.
No deben importar nada de este proyecto para evitar dependencias circulares.

── Cómo agregar o quitar un órgano ───────────────────────────────────────────
1. Editar la lista ORG_ORDER (agregar/quitar el nombre lógico del órgano).
2. Para cada entrada de PROTO_LIB, agregar/quitar la clave correspondiente
   en "B_ppm" y "B_err" (son dicts {organo: valor}, no listas — no hay orden
   que mantener a mano).
3. Agregar/quitar la clave correspondiente en BIO_LIB[preset]["RBE_value"] y
   BIO_LIB[preset]["CBE_value"] para cada preset (también dicts por nombre).
4. Si el órgano nuevo necesita un preset IsoE, agregarlo en
   ratmaster/physics/isoe.py (ISOE_PARAM_PRESETS) con "valid_organs"
   incluyendo el nombre nuevo.
5. Listo. Las tablas de la UI (boro, CBE/RBE, constraints) se generan
   automáticamente con las columnas de ORG_ORDER — no hace falta tocar
   ningún índice numérico en ningún diálogo.

No se necesita preservar ningún orden relativo entre estas estructuras:
todas se indexan por NOMBRE de órgano. Los arrays posicionales que usa
compute_bnct() internamente se construyen on-demand con organ_dict_to_array().

Protocolos/constraints YA GUARDADOS en disco (%APPDATA%\\RatMaster\\*.json)
de versiones anteriores (formato de listas posicionales) se migran
automáticamente la primera vez que se cargan — ver _migrate_boro_protocol_dict()
y _migrate_constraint_matrix() más abajo. La migración asume que la lista
vieja estaba alineada con el ORG_ORDER vigente *al momento de guardarla*;
si el archivo es de antes de este cambio, eso corresponde a los 8 órganos
originales: ["Pulmon", "Cerebro", "Medula", "Esofago", "Rinon", "Corazon",
"Tumor", "Piel"]. Esa lista se guarda como _LEGACY_ORG_ORDER exclusivamente
para poder migrar datos viejos; no se usa para nada más.
"""

import numpy as np

# ── Órganos y factor SPND ─────────────────────────────────────────────────────

# Orden canónico de órganos. Este orden se usa para generar las columnas de
# las tablas de la UI (boro, CBE/RBE, constraints) y el orden por defecto en
# el que se procesan los órganos. Para agregar o quitar un órgano, ver el
# instructivo al principio de este archivo.
ORG_ORDER = ["Pulmon", "Cerebro", "Medula", "Esofago", "Rinon", "Corazon", "Tumor", "Piel"]

# Orden de órganos vigente ANTES de este cambio (listas posicionales).
# Se usa SOLO para migrar protocolos/constraints guardados en disco por
# versiones anteriores de RatMaster, que no llevan nombre de órgano asociado
# a cada posición. No tocar aunque cambies ORG_ORDER en el futuro: es un
# valor histórico fijo, no la lista "actual".
_LEGACY_ORG_ORDER = ["Pulmon", "Cerebro", "Medula", "Esofago", "Rinon", "Corazon", "Tumor", "Piel"]

# Factor de corrección del flujo neutrónico para concordacia entre simulación y mediciones experimentales.
FLUX_CORR_FACTOR = 1.0 + 0.1263


# ── Paleta de colores para DVH (ciclo matplotlib extendido) ──────────────────

# Se asigna un color fijo por índice de órgano en el mapa generado al calcular.
# Orden coincide con el ciclo de color por defecto de matplotlib (tab10).
# Si hay más órganos que colores, se cicla (ver organ_color_for()).
ORGAN_COLORS = [
    "#1F77B4",  # azul
    "#FF7F0E",  # naranja
    "#2CA02C",  # verde
    "#D62728",  # rojo
    "#9467BD",  # violeta
    "#8C564B",  # marrón
    "#E377C2",  # rosa
    "#7F7F7F",  # gris
    "#BCBD22",  # oliva
    "#17BECF",  # cyan
]


def organ_color_for(index: int) -> str:
    """Color para el órgano en la posición `index` de ORG_ORDER, ciclando si hace falta."""
    if not ORGAN_COLORS:
        return "#7F7F7F"
    return ORGAN_COLORS[index % len(ORGAN_COLORS)]


# ── Helpers de conversión dict-por-nombre ⇄ array posicional ─────────────────
#
# Todas las cantidades por-órgano (B_ppm, B_err, RBE, CBE) se almacenan como
# dict {nombre_organo: valor}. compute_bnct() y el resto de la física siguen
# trabajando con arrays posicionales alineados a un organ_order dado — estos
# helpers hacen la conversión en el momento de usarlos, así nunca hay que
# mantener a mano la correspondencia índice↔nombre.

def organ_dict_to_array(d: dict, organ_order: list | None = None, default: float = 0.0) -> np.ndarray:
    """
    Convierte {organo: valor} a un array posicional alineado a organ_order
    (ORG_ORDER si no se especifica). Órganos de organ_order ausentes en `d`
    se completan con `default`. Claves de `d` que no están en organ_order
    se ignoran (no rompen nada: permite tener protocolos "más completos"
    que el ORG_ORDER vigente, por ejemplo si después se sacó un órgano).
    """
    order = list(organ_order or ORG_ORDER)
    d = d or {}
    return np.array([float(d.get(o, default)) for o in order], dtype=float)


def organ_array_to_dict(arr, organ_order: list | None = None) -> dict:
    """Convierte un array posicional alineado a organ_order a {organo: valor}."""
    order = list(organ_order or ORG_ORDER)
    arr = list(arr)
    return {o: float(arr[i]) for i, o in enumerate(order) if i < len(arr)}


# ── Presets de protocolos de boro ─────────────────────────────────────────────
#
# Estructura: { nombre_protocolo: { "ref": str, "B_ppm": {organo: valor},
#                                    "B_err": {organo: valor} } }
# B_ppm/B_err son dicts por NOMBRE de órgano — no hace falta que tengan
# exactamente las mismas claves que ORG_ORDER en todo momento: un órgano de
# ORG_ORDER sin entrada en el dict simplemente se completa con 0.0 al usarse
# (ver organ_dict_to_array). Para agregar/quitar un órgano de un protocolo,
# agregar/quitar su clave en estos dicts.

PROTO_LIB = {
    "BPA46.5": {
        "ref": "TODO: agregar referencia bibliográfica",
        # Concentración de boro tisular [ppm] por órgano (valores "genéricos",
        # es decir, los que se usan cuando NO se calcula a partir de sangre).
        "B_ppm": {
            "Pulmon": 12.2, "Cerebro": 5.5, "Medula": 5.5, "Esofago": 18.7,
            "Rinon": 69.0, "Corazon": 14.9, "Tumor": 22.9, "Piel": 18.5,
        },
        # Incertidumbre en B_ppm [ppm]
        "B_err": {
            "Pulmon": 7.2, "Cerebro": 2.4, "Medula": 2.4, "Esofago": 0.0,
            "Rinon": 32.0, "Corazon": 3.7, "Tumor": 7.2, "Piel": 7.9,
        },
        # Relación Tejido/Sangre (T/B): B_ppm_organo = TB_ratio[organo] × [B]_sangre.
        # Calculada como B_ppm/B_err de arriba dividido la concentración en
        # sangre de referencia del protocolo (ref_blood_conc = 13.7 ± 2.3 ppm),
        # propagando el error por cuadratura. Cerebro se asigna igual a Medula
        # (no hay dato de cerebro en la tabla; convención ya usada en B_ppm).
        # Esofago: TODO, la tabla no tiene una fila que se corresponda 1 a 1
        # (revisar si corresponde usar "Xiphoid" o "Costal muscle").
        "TB_ratio": {
            "Pulmon": 0.8905, "Cerebro": 0.4015, "Medula": 0.4015, "Esofago": 0.0,
            "Rinon": 5.0365, "Corazon": 1.0876, "Tumor": 1.6715, "Piel": 1.3504,
        },
        "TB_ratio_err": {
            "Pulmon": 0.5464, "Cerebro": 0.1877, "Medula": 0.1877, "Esofago": 0.0,
            "Rinon": 2.4911, "Corazon": 0.3261, "Tumor": 0.5958, "Piel": 0.6197,
        },
        # Concentración en sangre de referencia con la que se calculó TB_ratio
        # (solo informativo/trazabilidad; no se usa en el cálculo de dosis).
        "ref_blood_conc": 13.7, "ref_blood_conc_err": 2.3,
    },
    "B_GB10_50IV": {
        "ref": "TODO: agregar referencia bibliográfica",
        "B_ppm": {
            "Pulmon": 10.3, "Cerebro": 1.6, "Medula": 1.6, "Esofago": 0.0,
            "Rinon": 27.3, "Corazon": 7.2, "Tumor": 12.8, "Piel": 16.1,
        },
        "B_err": {
            "Pulmon": 2.4, "Cerebro": 0.8, "Medula": 0.8, "Esofago": 0.0,
            "Rinon": 7.3, "Corazon": 5.4, "Tumor": 4.1, "Piel": 8.7,
        },
        # Ídem BPA46.5: calculada con ref_blood_conc = 26.8 ± 14.2 ppm.
        # Esofago: TODO (sin fila correspondiente en la tabla).
        "TB_ratio": {
            "Pulmon": 0.3843, "Cerebro": 0.0597, "Medula": 0.0597, "Esofago": 0.0,
            "Rinon": 1.0187, "Corazon": 0.2687, "Tumor": 0.4776, "Piel": 0.6007,
        },
        "TB_ratio_err": {
            "Pulmon": 0.2225, "Cerebro": 0.0435, "Medula": 0.0435, "Esofago": 0.0,
            "Rinon": 0.6098, "Corazon": 0.1516, "Tumor": 0.2957, "Piel": 0.4546,
        },
        "ref_blood_conc": 26.8, "ref_blood_conc_err": 14.2,
    },
}

# Librería de presets de parámetros biológicos (CBE/RBE), por nombre — mismo
# esquema que PROTO_LIB para los protocolos de boro: cada entrada es un dict
# {"ref": str, "RBE_value": {organo: valor}, "CBE_value": {organo: valor}}.
# Se puede elegir un preset de acá o definir uno manual desde BioParamsDialog,
# y guardarlo como preset nuevo (queda persistido en disco, ver
# data/persistence.py y USER_BIO_PRESETS más abajo).
BIO_LIB = {
    # RBE=3.2 para todos los órganos, CBE=3.8 para tumor y 1.4 para el resto
    # de los tejidos — valores convencionales derivados de células de
    # glioblastoma, usados como default cuando no hay datos específicos del
    # tejido en cuestión.
    "Convencional (glioblastoma)": {
        "ref": "Valores convencionales (glioblastoma)",
        "RBE_value": {
            "Pulmon": 3.2, "Cerebro": 3.2, "Medula": 3.2, "Esofago": 3.2,
            "Rinon": 3.2, "Corazon": 3.2, "Tumor": 3.2, "Piel": 3.2,
        },
        "CBE_value": {
            "Pulmon": 1.4, "Cerebro": 1.4, "Medula": 1.4, "Esofago": 1.4,
            "Rinon": 1.4, "Corazon": 1.4, "Tumor": 3.8, "Piel": 1.4,
        },
    },
    # Valores calculados específicamente para Tumor, Pulmón y Piel; el resto
    # de los órganos usa los valores convencionales (3.2 / 1.4) por no tener
    # todavía un valor específico calculado.
    "Calculado (tejidos específicos)": {
        "ref": "TODO: agregar referencia bibliográfica",
        "RBE_value": {
            "Pulmon": 2.02, "Cerebro": 3.2, "Medula": 3.2, "Esofago": 3.2,
            "Rinon": 3.2, "Corazon": 3.2, "Tumor": 1.03, "Piel": 3.14,
        },
        "CBE_value": {
            "Pulmon": 2.31, "Cerebro": 1.4, "Medula": 1.4, "Esofago": 1.4,
            "Rinon": 1.4, "Corazon": 1.4, "Tumor": 3.35, "Piel": 3.76,
        },
    },
}

# Preset de BIO_LIB usado por defecto al arrancar la app o al cambiar de
# protocolo de boro desde el combo (ver load_defaults/apply_selected_protocol
# en ui/main_window.py).
DEFAULT_BIO_PRESET_NAME = "Convencional (glioblastoma)"

# Valor por defecto para un órgano nuevo que todavía no tiene entrada en
# RBE_value / CBE_value (p. ej. recién agregado a ORG_ORDER). 1.0 = sin
# efecto biológico relativo adicional; conviene revisarlo manualmente.
DEFAULT_RBE_FOR_NEW_ORGAN = 1.0
DEFAULT_CBE_FOR_NEW_ORGAN = 1.0

# Dictionaries para protocolos, constraints e IsoE definidos por el usuario.
# Se poblan al iniciar la app llamando a load_all_user_data() (ver abajo).
USER_BORO_PROTOCOLS: dict = {}
USER_CONSTRAINT_PRESETS: dict = {}
USER_ISOE_PRESETS: dict = {}
USER_BIO_PRESETS: dict = {}


# ── Presets de constraints (matrices 5×N_organs) ──────────────────────────────
#
# Cada preset es una matriz numpy de shape (5, len(ORG_ORDER)):
#   fila 0 → Dmax permitida por órgano
#   fila 1 → Dmean permitida
#   fila 2 → Dmin requerida (solo tiene sentido para Tumor)
#   fila 3 → Vx% (porcentaje de volumen para constraint Dose@Vx)
#   fila 4 → Dose@Vx (dosis en ese volumen)
# Un valor 0 significa "sin restricción".

def _make_constraints_matrix(default: float = 0.0) -> np.ndarray:
    """Crea una matriz de constraints vacía (todos en 0 = sin restricción)."""
    return np.full((5, len(ORG_ORDER)), float(default), dtype=float)


CONSTRAINT_PRESETS: dict[str, np.ndarray] = {}

# Preset de ejemplo: solo fija Dmin en Tumor = 8 Gy
_m = _make_constraints_matrix(0.0)
try:
    _m[2, ORG_ORDER.index("Tumor")] = 8.0  # fila 2 = Dmin
except Exception:
    pass
CONSTRAINT_PRESETS["Tumor Dmin 8 Gy"] = _m


# ── Helpers de validación y serialización ────────────────────────────────────

def _make_constraints_matrix(default: float = 0.0) -> np.ndarray:
    """Crea una matriz de constraints vacía (todos 0 = sin restricción)."""
    return np.full((5, len(ORG_ORDER)), float(default), dtype=float)


def _constraints_matrix_to_dict(mat, organ_order: list | None = None) -> dict:
    """
    Convierte una matriz de constraints (5×N) a formato serializable por
    nombre de órgano: {organo: [Dmax, Dmean, Dmin, Vx, Dx]}. Este es el
    formato que se guarda en disco a partir de ahora (en vez de una lista
    de listas posicional), para que agregar/quitar órganos no desalinee
    los constraints ya guardados.
    """
    order = list(organ_order or ORG_ORDER)
    arr = np.asarray(mat, dtype=float)
    return {
        organ: arr[:, j].tolist()
        for j, organ in enumerate(order)
        if j < arr.shape[1]
    }


def _constraints_matrix_from_dict(d: dict, organ_order: list | None = None) -> np.ndarray:
    """
    Reconstruye la matriz (5×len(organ_order)) desde el formato por nombre
    {organo: [Dmax, Dmean, Dmin, Vx, Dx]}. Órganos de organ_order ausentes
    en `d` quedan en 0 (sin restricción). Claves de `d` que ya no están en
    organ_order se ignoran.
    """
    order = list(organ_order or ORG_ORDER)
    mat = np.zeros((5, len(order)), dtype=float)
    for j, organ in enumerate(order):
        row = d.get(organ)
        if row is None:
            continue
        row = list(row)[:5]
        mat[:len(row), j] = row
    return mat


def _constraints_matrix_from_serializable(obj, organ_order: list | None = None) -> np.ndarray:
    """
    Convierte datos cargados de JSON a matriz numpy (5, len(organ_order)).

    Soporta dos formatos:
      - NUEVO (por nombre):  {"Pulmon": [Dmax,Dmean,Dmin,Vx,Dx], "Cerebro": [...], ...}
        Robusto a agregar/quitar órganos: cada fila se ubica por nombre.
      - VIEJO (posicional):  [[fila0...], [fila1...], ..., [fila4...]]
        Lista de 5 listas de longitud N. Se asume alineada a _LEGACY_ORG_ORDER
        (8 órganos originales) y se migra automáticamente a las columnas de
        organ_order por nombre. Si la cantidad de columnas no coincide con
        _LEGACY_ORG_ORDER, se rechaza (no hay forma segura de adivinar a qué
        órgano corresponde cada columna).

    Lanza ValueError si el formato no se reconoce o es inválido.
    """
    order = list(organ_order or ORG_ORDER)

    if isinstance(obj, dict):
        return _constraints_matrix_from_dict(obj, order)

    # Formato viejo: lista de listas posicional
    arr = np.array(obj, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 5:
        raise ValueError(
            f"Matriz de constraints inválida: se esperaba (5, N), se recibió {arr.shape}"
        )
    if arr.shape[1] == len(_LEGACY_ORG_ORDER):
        # Migración: la columna j corresponde a _LEGACY_ORG_ORDER[j]
        as_dict = {organ: arr[:, j].tolist() for j, organ in enumerate(_LEGACY_ORG_ORDER)}
        return _constraints_matrix_from_dict(as_dict, order)
    if arr.shape[1] == len(order):
        # Mismo tamaño que el ORG_ORDER actual: se asume ya alineado
        # (caso límite — solo pasa si nunca se migró y casualmente coincide
        # la cantidad de órganos). Más seguro que rechazar de plano.
        return arr
    raise ValueError(
        f"Matriz de constraints en formato posicional antiguo con "
        f"{arr.shape[1]} columnas: no coincide ni con el orden legacy "
        f"({len(_LEGACY_ORG_ORDER)} órganos) ni con ORG_ORDER actual "
        f"({len(order)} órganos). No se puede migrar de forma segura."
    )


def _sanitize_boro_protocol_dict(name: str, data: dict, organ_order: list | None = None) -> dict:
    """
    Valida y normaliza un dict de protocolo de boro definido por el usuario.

    Soporta dos formatos de entrada para B_ppm/B_err:
      - NUEVO (por nombre): {"Pulmon": 12.2, "Cerebro": 5.5, ...}
      - VIEJO (posicional): [12.2, 5.5, ...] alineado a _LEGACY_ORG_ORDER
        (se migra automáticamente a dict por nombre).

    El resultado siempre tiene B_ppm/B_err como dict {organo: valor}.
    Órganos de organ_order sin valor en la entrada quedan implícitos en 0.0
    al usarse (ver organ_dict_to_array) — no hace falta que el protocolo
    tenga TODOS los órganos de organ_order para ser válido.

    Lanza ValueError si hay valores negativos o el formato no se reconoce.
    """
    order = list(organ_order or ORG_ORDER)

    def _to_dict(raw, label):
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {str(k): float(v) for k, v in raw.items()}
        # Formato viejo: lista posicional alineada a _LEGACY_ORG_ORDER
        arr = np.array(raw, dtype=float).ravel()
        if arr.size == len(_LEGACY_ORG_ORDER):
            return {organ: float(arr[i]) for i, organ in enumerate(_LEGACY_ORG_ORDER)}
        if arr.size == len(order):
            return {organ: float(arr[i]) for i, organ in enumerate(order)}
        raise ValueError(
            f"Protocolo '{name}': '{label}' tiene {arr.size} valores en formato "
            f"posicional antiguo, no coincide ni con el orden legacy "
            f"({len(_LEGACY_ORG_ORDER)} órganos) ni con el orden actual "
            f"({len(order)} órganos). No se puede migrar de forma segura."
        )

    b_dict  = _to_dict(data.get("B_ppm"), "B_ppm")
    be_dict = _to_dict(data.get("B_err"), "B_err")

    if any(v < 0 for v in b_dict.values()) or any(v < 0 for v in be_dict.values()):
        raise ValueError(f"Protocolo '{name}' tiene valores negativos.")

    # Relación Tejido/Sangre (TB_ratio/TB_ratio_err): campo nuevo, siempre en
    # formato dict {organo: valor} — no existe versión posicional vieja para
    # migrar, porque no existía antes de esta funcionalidad. Ausente = {}
    # (protocolo sin datos de sangre todavía).
    tb_raw  = data.get("TB_ratio") or {}
    tbe_raw = data.get("TB_ratio_err") or {}
    tb_dict  = {str(k): float(v) for k, v in tb_raw.items()} if isinstance(tb_raw, dict) else {}
    tbe_dict = {str(k): float(v) for k, v in tbe_raw.items()} if isinstance(tbe_raw, dict) else {}
    if any(v < 0 for v in tb_dict.values()) or any(v < 0 for v in tbe_dict.values()):
        raise ValueError(f"Protocolo '{name}' tiene valores negativos en la relación Tejido/Sangre.")

    # Concentración en sangre de referencia usada para derivar TB_ratio (solo
    # informativa/trazabilidad, no interviene en el cálculo de dosis).
    ref_blood = float(data.get("ref_blood_conc", 0.0) or 0.0)
    ref_blood_err = float(data.get("ref_blood_conc_err", 0.0) or 0.0)

    return {
        "ref": data.get("ref", "Usuario"),
        "B_ppm": b_dict, "B_err": be_dict,
        "TB_ratio": tb_dict, "TB_ratio_err": tbe_dict,
        "ref_blood_conc": ref_blood, "ref_blood_conc_err": ref_blood_err,
    }


def _is_builtin_boro_protocol(name: str) -> bool:
    """True si el nombre corresponde a un protocolo de boro incluido en PROTO_LIB."""
    return name in {"BPA46.5", "B_GB10_50IV"}


def protocol_has_blood_ratio(name: str) -> bool:
    """
    True si el protocolo `name` tiene al menos una relación Tejido/Sangre
    (TB_ratio) cargada, es decir, se puede usar el modo "a partir de sangre"
    con este protocolo sin tener que cargar los valores a mano.
    """
    p = PROTO_LIB.get(name) or {}
    tb = p.get("TB_ratio") or {}
    return any(float(v) != 0.0 for v in tb.values())


def _sanitize_bio_preset_dict(name: str, data: dict, organ_order: list | None = None) -> dict:
    """
    Valida y normaliza un dict de preset de parámetros biológicos (CBE/RBE)
    definido por el usuario. Mismo esquema que _sanitize_boro_protocol_dict,
    pero para RBE_value/CBE_value en vez de B_ppm/B_err (sin formato
    posicional viejo, porque no existía antes de esta funcionalidad).

    Lanza ValueError si hay valores negativos.
    """
    def _to_dict(raw):
        raw = raw or {}
        return {str(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

    rbe_dict = _to_dict(data.get("RBE_value"))
    cbe_dict = _to_dict(data.get("CBE_value"))
    if any(v < 0 for v in rbe_dict.values()) or any(v < 0 for v in cbe_dict.values()):
        raise ValueError(f"Preset biológico '{name}' tiene valores negativos.")

    return {
        "ref": data.get("ref", "Usuario"),
        "RBE_value": rbe_dict, "CBE_value": cbe_dict,
    }


def _is_builtin_bio_preset(name: str) -> bool:
    """True si el nombre corresponde a un preset de CBE/RBE incluido en BIO_LIB."""
    return name in BIO_LIB and name in {
        "Convencional (glioblastoma)", "Calculado (tejidos específicos)",
    }


def _is_builtin_isoe_preset(name: str) -> bool:
    """
    True si el preset IsoE es uno de los incorporados (no de usuario).

    [FIX] Antes este set tenía nombres ("Gonzalez2012_GS9L_synergy",
    "Gonzalez2012_MelJ_synergy") que ya no existen en ISOE_PARAM_PRESETS
    (quedaron de un renombrado anterior). Como consecuencia, si un usuario
    guardaba un preset propio con el mismo nombre que un preset builtin real
    (p. ej. "Tumor_DHD" o "Medula_Dattoli2025"), load_all_user_data() no lo
    reconocía como protegido y lo sobreescribía en ISOE_PARAM_PRESETS sin
    avisar. Ahora se deriva dinámicamente de isoe.ISOE_PARAM_PRESETS para
    que nunca pueda desincronizarse de nuevo.
    """
    try:
        from ratmaster.physics.isoe import ISOE_PARAM_PRESETS
        return name in ISOE_PARAM_PRESETS
    except Exception:
        # Fallback si physics.isoe todavía no se pudo importar (no debería
        # pasar en uso normal, pero evita un crash duro al arrancar).
        return name == "Manual"


def _is_builtin_constraint_preset(name: str) -> bool:
    """True si el preset de constraints es el incorporado por defecto."""
    return name in {"Tumor Dmin 8 Gy"}


def _resolve_boro_protocol_name(raw_name: str) -> str:
    """
    Normaliza el nombre de un protocolo de boro.
    'No especificado', cadena vacía y '(manual)' se resuelven a 'Manual'.
    """
    name = str(raw_name or "").strip()
    if not name or name.lower() == "no especificado" or name == "(manual)":
        return "Manual"
    return name


def _defaults_from_libs(proto_name: str = "BPA46.5", organ_order: list | None = None,
                         bio_name: str = "") -> dict:
    """
    Construye un dict de defaults (B_arr, B_err, CBE, RBE, Constraints)
    a partir de las librerías incorporadas PROTO_LIB (boro) y BIO_LIB (CBE/RBE).
    Compatible con el formato esperado por load_defaults() en la ventana principal.

    B_ppm/B_err/CBE_value/RBE_value son dicts por nombre de órgano: un órgano
    de organ_order sin entrada en el preset se completa con 0.0 (boro) o 1.0
    (CBE/RBE, ver DEFAULT_CBE_FOR_NEW_ORGAN / DEFAULT_RBE_FOR_NEW_ORGAN) en
    vez de fallar — así un órgano agregado recientemente a ORG_ORDER no
    rompe los protocolos preexistentes, aunque sí conviene completarlo a mano.

    bio_name: nombre del preset de BIO_LIB a usar para CBE/RBE. Si no se
    especifica o no existe, se usa DEFAULT_BIO_PRESET_NAME.
    """
    order = list(organ_order or ORG_ORDER)
    if proto_name not in PROTO_LIB:
        proto_name = "BPA46.5"
    p = PROTO_LIB[proto_name]
    if bio_name not in BIO_LIB:
        bio_name = DEFAULT_BIO_PRESET_NAME
    bio = BIO_LIB[bio_name]
    return {
        "B_arr": organ_dict_to_array(p["B_ppm"], order, default=0.0),
        "B_err": organ_dict_to_array(p["B_err"], order, default=0.0),
        # Relación Tejido/Sangre del protocolo (0.0 = sin dato para ese órgano).
        "TB_ratio":     organ_dict_to_array(p.get("TB_ratio") or {}, order, default=0.0),
        "TB_ratio_err": organ_dict_to_array(p.get("TB_ratio_err") or {}, order, default=0.0),
        "CBE":   organ_dict_to_array(bio["CBE_value"], order, default=DEFAULT_CBE_FOR_NEW_ORGAN),
        "RBE":   organ_dict_to_array(bio["RBE_value"], order, default=DEFAULT_RBE_FOR_NEW_ORGAN),
        "Constraints": None,
    }


# ── Carga de datos de usuario al iniciar ─────────────────────────────────────

def load_all_user_data() -> None:
    """
    Carga desde disco todos los datos persistidos por el usuario y los
    incorpora en los dicts globales de este módulo (PROTO_LIB, BIO_LIB,
    CONSTRAINT_PRESETS, USER_BORO_PROTOCOLS, USER_CONSTRAINT_PRESETS,
    USER_ISOE_PRESETS, USER_BIO_PRESETS).

    Debe llamarse UNA sola vez al arrancar la aplicación, antes de que
    se cree cualquier ventana o diálogo.

    Los presets builtin nunca se sobreescriben: si hay un nombre de usuario
    que coincide con un builtin, la entrada de usuario se ignora.

    Migración automática: protocolos de boro y constraints guardados por
    versiones anteriores de RatMaster (formato de listas posicionales,
    sin nombre de órgano) se convierten al formato por nombre la primera
    vez que se cargan, alineándolos contra _LEGACY_ORG_ORDER. Si una
    entrada no se puede migrar de forma segura (cantidad de valores no
    coincide ni con el orden legacy ni con ORG_ORDER actual), se omite y
    se informa por consola — no se pierde silenciosamente sin dejar rastro,
    pero tampoco bloquea el arranque de la app.
    """
    # Importación local para evitar dependencia circular en module-level
    from ratmaster.data.persistence import (
        load_user_boro_protocols,
        load_user_constraint_presets,
        load_user_isoe_presets,
        load_user_bio_presets,
    )

    # ── Protocolos de boro ────────────────────────────────────────────────────
    raw_boro = load_user_boro_protocols()
    for name, data in raw_boro.items():
        if _is_builtin_boro_protocol(name):
            continue
        try:
            clean = _sanitize_boro_protocol_dict(name, data, ORG_ORDER)
            PROTO_LIB[name]           = clean
            USER_BORO_PROTOCOLS[name] = clean
        except Exception as e:
            print(f"[RatMaster] Protocolo de boro '{name}' omitido (no se pudo cargar/migrar): {e}")

    # ── Constraints ───────────────────────────────────────────────────────────
    # raw_cons ya viene como np.ndarray (5, N) desde load_user_constraint_presets(),
    # que internamente usa _constraints_matrix_from_serializable() y por lo tanto
    # ya migra el formato viejo. Si la migración falló para alguna entrada,
    # load_user_constraint_presets() ya la omitió (ver data/persistence.py).
    raw_cons = load_user_constraint_presets()
    for name, mat in raw_cons.items():
        if _is_builtin_constraint_preset(name):
            continue
        CONSTRAINT_PRESETS[name]       = mat
        USER_CONSTRAINT_PRESETS[name]  = mat

    # ── Presets IsoE ──────────────────────────────────────────────────────────
    raw_isoe = load_user_isoe_presets()
    for name, preset in raw_isoe.items():
        if _is_builtin_isoe_preset(name):
            continue
        if not isinstance(preset, dict):
            continue
        # Incorporar en el dict global de physics/isoe (si ya fue importado)
        try:
            from ratmaster.physics.isoe import ISOE_PARAM_PRESETS
            ISOE_PARAM_PRESETS[name] = preset
        except Exception:
            pass
        USER_ISOE_PRESETS[name] = preset

    # ── Presets biológicos (CBE/RBE) ─────────────────────────────────────────
    raw_bio = load_user_bio_presets()
    for name, data in raw_bio.items():
        if _is_builtin_bio_preset(name):
            continue
        try:
            clean = _sanitize_bio_preset_dict(name, data, ORG_ORDER)
            BIO_LIB[name]          = clean
            USER_BIO_PRESETS[name] = clean
        except Exception as e:
            print(f"[RatMaster] Preset biológico '{name}' omitido (no se pudo cargar): {e}")
