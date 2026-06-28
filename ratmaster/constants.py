"""
constants.py
============
Constantes globales de RatMaster: orden de órganos, factores de calibración,
paleta de colores para DVH, y presets de protocolos de boro, RBE, CBE y constraints.

Estas definiciones son compartidas por los módulos de física, datos y UI.
No deben importar nada de este proyecto para evitar dependencias circulares.
"""

import numpy as np

# ── Órganos y factor SPND ─────────────────────────────────────────────────────

# Orden canónico de órganos. Este orden se usa en todas las tablas y matrices.
ORG_ORDER = ["Pulmon", "Cerebro", "Medula", "Esofago", "Rinon", "Corazon", "Tumor", "Piel"]

# Factor de corrección del flujo neutrónico para concordacia entre simulación y mediciones experimentales.
FLUX_CORR_FACTOR = 1.0 + 0.1263


# ── Paleta de colores para DVH (ciclo matplotlib extendido) ──────────────────

# Se asigna un color fijo por índice de órgano en el mapa generado al calcular.
# Orden coincide con el ciclo de color por defecto de matplotlib (tab10).
ORGAN_COLORS = [
    "#1F77B4",  # azul
    "#FF7F0E",  # naranja
    "#2CA02C",  # verde
    "#D62728",  # rojo
    "#9467BD",  # violeta
    "#8C564B",  # marrón
    "#E377C2",  # rosa
    "#7F7F7F",  # gris
]


# ── Presets de protocolos de boro ─────────────────────────────────────────────
#
# Estructura: { nombre_protocolo: { "ref": str, "B_ppm": list, "B_err": list } }
# Longitud de las listas = len(ORG_ORDER). Orden: Pulmon, Cerebro, ..., Piel.

PROTO_LIB = {
    "BPA46.5": {
        "ref": "TODO: agregar referencia bibliográfica",
        # Concentración de boro tisular [ppm] por órgano
        "B_ppm": [12.2, 5.5,  5.5,  18.7, 69.0, 14.9, 22.9, 18.5],
        # Incertidumbre en B_ppm [ppm]
        "B_err": [ 7.2, 2.4,  2.4,   0.0, 32.0,  3.7,  7.2,  7.9],
    },
    "B_GB10_50IV": {
        "ref": "TODO: agregar referencia bibliográfica",
        "B_ppm": [10.3, 1.6, 1.6,  0.0, 27.3, 7.2, 12.8, 16.1],
        "B_err": [ 2.4, 0.8, 0.8,  0.0,  7.3, 5.4,  4.1,  8.7],
    },
}

# Valores de RBE por órgano (mismo orden que ORG_ORDER)
RBE_PRESET = {
    "ref": "TODO: agregar referencia bibliográfica",
    #["Pulmon", "Cerebro", "Medula", "Esofago", "Rinon", "Corazon", "Tumor", "Piel"]
    #"RBE_value": [2.02, 2.9, 2.9, 3.2, 3.2, 3.2, 1.03, 3.14],
    "RBE_value": [3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2],
}

# Valores de CBE por órgano (mismo orden que ORG_ORDER)
# Nota: el Tumor tiene CBE mayor (3.8) que los tejidos normales (1.4)
CBE_PRESET = {
    "ref": "TODO: agregar referencia bibliográfica",
    #"CBE_value": [2.31, 2.2 ,  2.2 , 1.4, 1.4, 1.4, 3.35, 3.76],
    "CBE_value": [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 3.8, 1.4],
}

# Dictionaries para protocolos, constraints e IsoE definidos por el usuario.
# Se poblan al iniciar la app llamando a load_all_user_data() (ver abajo).
USER_BORO_PROTOCOLS: dict = {}
USER_CONSTRAINT_PRESETS: dict = {}
USER_ISOE_PRESETS: dict = {}


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


def _constraints_matrix_from_serializable(obj) -> np.ndarray:
    """
    Convierte una lista de listas (cargada de JSON) a matriz numpy validada.
    Lanza ValueError si la forma no es (5, N_organs).
    """
    arr = np.array(obj, dtype=float)
    if arr.shape != (5, len(ORG_ORDER)):
        raise ValueError(
            f"Matriz de constraints inválida: se esperaba (5, {len(ORG_ORDER)}), "
            f"se recibió {arr.shape}"
        )
    return arr


def _sanitize_boro_protocol_dict(name: str, data: dict) -> dict:
    """
    Valida y normaliza un dict de protocolo de boro definido por el usuario.
    Lanza ValueError si la longitud no coincide con ORG_ORDER o hay valores negativos.
    """
    b  = np.array(data.get("B_ppm", []), dtype=float).ravel()
    be = np.array(data.get("B_err", []), dtype=float).ravel()
    if b.size != len(ORG_ORDER) or be.size != len(ORG_ORDER):
        raise ValueError(
            f"Protocolo '{name}': se esperan {len(ORG_ORDER)} valores, "
            f"se recibieron {b.size} (B_ppm) y {be.size} (B_err)."
        )
    if np.any(b < 0) or np.any(be < 0):
        raise ValueError(f"Protocolo '{name}' tiene valores negativos.")
    return {"ref": data.get("ref", "Usuario"), "B_ppm": b.tolist(), "B_err": be.tolist()}


def _is_builtin_boro_protocol(name: str) -> bool:
    """True si el nombre corresponde a un protocolo de boro incluido en PROTO_LIB."""
    return name in {"BPA46.5", "B_GB10_50IV"}


def _is_builtin_isoe_preset(name: str) -> bool:
    """True si el preset IsoE es uno de los incorporados (no de usuario)."""
    return name in {"Manual", "Gonzalez2012_GS9L_synergy", "Gonzalez2012_MelJ_synergy"}


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


def _defaults_from_libs(proto_name: str = "BPA46.5") -> dict:
    """
    Construye un dict de defaults (B_arr, B_err, CBE, RBE, Constraints)
    a partir de las librerías incorporadas PROTO_LIB / RBE_PRESET / CBE_PRESET.
    Compatible con el formato esperado por load_defaults() en la ventana principal.
    """
    if proto_name not in PROTO_LIB:
        proto_name = "BPA46.5"
    p = PROTO_LIB[proto_name]
    return {
        "B_arr":       np.array(p["B_ppm"],              dtype=float).ravel(),
        "B_err":       np.array(p["B_err"],              dtype=float).ravel(),
        "CBE":         np.array(CBE_PRESET["CBE_value"], dtype=float).ravel(),
        "RBE":         np.array(RBE_PRESET["RBE_value"], dtype=float).ravel(),
        "Constraints": None,
    }


# ── Carga de datos de usuario al iniciar ─────────────────────────────────────

def load_all_user_data() -> None:
    """
    Carga desde disco todos los datos persistidos por el usuario y los
    incorpora en los dicts globales de este módulo (PROTO_LIB,
    CONSTRAINT_PRESETS, USER_BORO_PROTOCOLS, USER_CONSTRAINT_PRESETS,
    USER_ISOE_PRESETS).

    Debe llamarse UNA sola vez al arrancar la aplicación, antes de que
    se cree cualquier ventana o diálogo.

    Los presets builtin nunca se sobreescriben: si hay un nombre de usuario
    que coincide con un builtin, la entrada de usuario se ignora.
    """
    # Importación local para evitar dependencia circular en module-level
    from ratmaster.data.persistence import (
        load_user_boro_protocols,
        load_user_constraint_presets,
        load_user_isoe_presets,
    )

    # ── Protocolos de boro ────────────────────────────────────────────────────
    raw_boro = load_user_boro_protocols()
    for name, data in raw_boro.items():
        if _is_builtin_boro_protocol(name):
            continue
        try:
            clean = _sanitize_boro_protocol_dict(name, data)
            PROTO_LIB[name]           = clean
            USER_BORO_PROTOCOLS[name] = clean
        except Exception:
            pass  # entrada corrupta: se omite silenciosamente

    # ── Constraints ───────────────────────────────────────────────────────────
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
