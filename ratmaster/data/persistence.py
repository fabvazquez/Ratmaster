"""
data/persistence.py
===================
Persistencia de configuración y datos de usuario de RatMaster.

Incluye:
  - DEFAULT_SPND_REGISTRY: detectores SPND con factores de calibración y sensibilidad.
  - load_spnd_registry() / save_spnd_registry(): lectura/escritura del registry JSON.
  - load_user_boro_protocols() / save_user_boro_protocols(): protocolos de boro del usuario.
  - load_user_constraint_presets() / save_user_constraint_presets(): constraints del usuario.
  - load_user_isoe_presets() / save_user_isoe_presets(): presets IsoE del usuario.
  - load_user_bio_presets() / save_user_bio_presets(): presets de CBE/RBE del usuario.
  - parse_number_or_pair(): parseo robusto de "4.32 ± 0.01" desde entrada de usuario.

Los archivos JSON se guardan en %APPDATA%\\RatMaster\\ (ver app_paths.py).
"""

import json
import numpy as np
from ratmaster.app_paths import ensure_user_json, user_writable_path


# ── Registry de detectores SPND ───────────────────────────────────────────────
#
# Estructura de cada detector:
#   name:             nombre identificatorio (ej: "Rojo", "Verde (ref)")
#   factor_to_verde:  factor de normalización respecto al detector de referencia
#   sens:             sensibilidad [A/(n/cm²/s)], None si no calibrado
#   sens_sigma:       incertidumbre en sensibilidad, None si no calibrado
#   last_calib:       fecha de última calibración (string ISO o "")
#   notes:            notas libres

DEFAULT_SPND_REGISTRY: dict = {
    "version": 3,
    "updated_label": "Actualizados a may 2024",
    "units": {
        "current":     "pA",
        "sensitivity": "A/(n/cm^2/s)",
    },
    "detectors": [
        {
            "name":            "Rojo",
            "factor_to_verde": 1.02541600,
            "sens":            1.90e-21,
            "sens_sigma":      1.50e-22,
            "last_calib":      "",
            "notes":           "",
        },
        {
            "name":            "A0",
            "factor_to_verde": 0.98599951,
            "sens":            None,
            "sens_sigma":      None,
            "last_calib":      "",
            "notes":           "",
        },
        {
            "name":            "A1",
            "factor_to_verde": 1.00603069,
            "sens":            None,
            "sens_sigma":      None,
            "last_calib":      "",
            "notes":           "",
        },
        {
            "name":            "Verde (ref)",
            "factor_to_verde": 1.0,
            "sens":            1.98e-21,
            "sens_sigma":      1.60e-22,
            "last_calib":      "",
            "notes":           "",
        },
    ],
}


# ── Acceso al registry SPND ───────────────────────────────────────────────────

def _registry_path():
    """Ruta al JSON del registry SPND en el directorio del usuario."""
    return ensure_user_json("spnd_registry.json", DEFAULT_SPND_REGISTRY)


def load_spnd_registry() -> dict:
    """
    Carga el registry de detectores SPND desde el JSON del usuario.
    Si el archivo no existe o está corrupto, devuelve el registry por defecto.
    """
    p = _registry_path()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "detectors" in data and isinstance(data["detectors"], list):
                return data
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_SPND_REGISTRY))


def save_spnd_registry(reg: dict) -> tuple[bool, str]:
    """
    Guarda el registry de detectores SPND en el JSON del usuario.

    Returns:
        (True, "") si fue exitoso.
        (False, mensaje_de_error) si falló.
    """
    p = _registry_path()
    try:
        p.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Protocolos de boro del usuario ────────────────────────────────────────────
#
# Archivo: %APPDATA%\RatMaster\user_boro_protocols.json
# Formato (desde este cambio): { "nombre": { "ref": str,
#     "B_ppm": {"Pulmon": 12.2, "Cerebro": 5.5, ...},
#     "B_err": {"Pulmon": 7.2, "Cerebro": 2.4, ...} }, ... }
# (por nombre de órgano — robusto a agregar/quitar órganos de ORG_ORDER).
#
# Archivos de versiones anteriores pueden tener B_ppm/B_err como listas
# posicionales (alineadas al ORG_ORDER de 8 órganos vigente en ese momento).
# Se migran automáticamente al cargar — ver constants._sanitize_boro_protocol_dict().

_BORO_FILE = "user_boro_protocols.json"


def load_user_boro_protocols() -> dict:
    """
    Carga los protocolos de boro definidos por el usuario desde el JSON del usuario.
    Devuelve un dict vacío si el archivo no existe o está corrupto.
    """
    p = ensure_user_json(_BORO_FILE, {})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_user_boro_protocols(protocols: dict) -> tuple[bool, str]:
    """
    Guarda el dict completo de protocolos de boro del usuario en disco.

    Args:
        protocols: dict con todos los protocolos de usuario
                   (solo los de usuario, no los builtin).

    Returns:
        (True, "") si fue exitoso.
        (False, mensaje_de_error) si falló.
    """
    p = user_writable_path(_BORO_FILE)
    try:
        p.write_text(json.dumps(protocols, indent=2, ensure_ascii=False), encoding="utf-8")
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Presets de constraints del usuario ───────────────────────────────────────
#
# Archivo: %APPDATA%\RatMaster\user_constraint_presets.json
# Formato (desde este cambio): { "nombre": {"Pulmon": [Dmax,Dmean,Dmin,Vx,Dx],
#                                            "Cerebro": [...], ...}, ... }
# (por nombre de órgano — robusto a agregar/quitar órganos de ORG_ORDER).
#
# Archivos guardados por versiones ANTERIORES de RatMaster pueden tener el
# formato viejo: { "nombre": [[fila0...], [fila1...], ..., [fila4...]] }
# (lista de listas posicional, alineada al ORG_ORDER de 8 órganos vigente en
# ese momento). Esos archivos se migran automáticamente al cargar — ver
# constants._constraints_matrix_from_serializable().

_CONSTRAINTS_FILE = "user_constraint_presets.json"


def load_user_constraint_presets() -> dict[str, np.ndarray]:
    """
    Carga los presets de constraints definidos por el usuario.
    Devuelve un dict {nombre: np.ndarray shape (5, len(ORG_ORDER))}.

    Migra automáticamente entradas en el formato posicional antiguo (ver
    constants._constraints_matrix_from_serializable). Entradas que no se
    puedan migrar de forma segura (cantidad de columnas no reconocida) se
    omiten, dejando un mensaje en consola en vez de fallar en silencio.
    """
    from ratmaster.constants import _constraints_matrix_from_serializable
    p = ensure_user_json(_CONSTRAINTS_FILE, {})
    result: dict[str, np.ndarray] = {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return result
        for name, raw in data.items():
            try:
                result[name] = _constraints_matrix_from_serializable(raw)
            except Exception as e:
                print(f"[RatMaster] Preset de constraints '{name}' omitido (no se pudo cargar/migrar): {e}")
    except Exception:
        pass
    return result


def save_user_constraint_presets(presets: dict[str, np.ndarray]) -> tuple[bool, str]:
    """
    Guarda el dict completo de presets de constraints del usuario en disco.

    Se serializa en el formato NUEVO por nombre de órgano (ver módulo),
    usando el ORG_ORDER vigente al momento de guardar. Esto hace que, si
    más adelante se agrega o quita un órgano de ORG_ORDER, este archivo ya
    guardado siga siendo válido (los órganos en común se leen por nombre;
    los que ya no existen en ORG_ORDER simplemente se ignoran al cargar).

    Returns:
        (True, "") si fue exitoso.
        (False, mensaje_de_error) si falló.
    """
    from ratmaster.constants import _constraints_matrix_to_dict, ORG_ORDER
    p = user_writable_path(_CONSTRAINTS_FILE)
    try:
        serializable = {
            name: _constraints_matrix_to_dict(mat, ORG_ORDER)
            for name, mat in presets.items()
            if isinstance(mat, np.ndarray)
        }
        p.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Presets IsoE del usuario ──────────────────────────────────────────────────
#
# Archivo: %APPDATA%\RatMaster\user_isoe_presets.json
# Formato: mismo que ISOE_PARAM_PRESETS en physics/isoe.py
# (dicts con "params", "t0_map", "tissue_type", "ref", etc.)

_ISOE_FILE = "user_isoe_presets.json"


def load_user_isoe_presets() -> dict:
    """
    Carga los presets IsoE definidos por el usuario.
    Devuelve un dict vacío si el archivo no existe o está corrupto.
    """
    p = ensure_user_json(_ISOE_FILE, {})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_user_isoe_presets(presets: dict) -> tuple[bool, str]:
    """
    Guarda el dict completo de presets IsoE del usuario en disco.

    Returns:
        (True, "") si fue exitoso.
        (False, mensaje_de_error) si falló.
    """
    p = user_writable_path(_ISOE_FILE)
    try:
        p.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Presets biológicos (CBE/RBE) del usuario ─────────────────────────────────
#
# Archivo: %APPDATA%\RatMaster\user_bio_presets.json
# Formato: { "nombre": {"ref": str,
#     "RBE_value": {"Pulmon": 3.2, "Cerebro": 3.2, ...},
#     "CBE_value": {"Pulmon": 1.4, "Cerebro": 1.4, ...} }, ... }
# (por nombre de órgano, igual que los protocolos de boro).

_BIO_FILE = "user_bio_presets.json"


def load_user_bio_presets() -> dict:
    """
    Carga los presets biológicos (CBE/RBE) definidos por el usuario desde el
    JSON del usuario. Devuelve un dict vacío si el archivo no existe o está
    corrupto.
    """
    p = ensure_user_json(_BIO_FILE, {})
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_user_bio_presets(presets: dict) -> tuple[bool, str]:
    """
    Guarda el dict completo de presets biológicos (CBE/RBE) del usuario en disco.

    Args:
        presets: dict con todos los presets de usuario (solo los de usuario,
                 no los builtin de BIO_LIB).

    Returns:
        (True, "") si fue exitoso.
        (False, mensaje_de_error) si falló.
    """
    p = user_writable_path(_BIO_FILE)
    try:
        p.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Parseo de números con incertidumbre ───────────────────────────────────────

def _normalize_num_text(s: str) -> str:
    """
    Normaliza un string numérico para parsearlo con float():
    - Convierte coma decimal → punto.
    - Convierte guión largo/em-dash → guión ASCII.
    """
    s = (s or "").strip()
    s = s.replace("−", "-").replace("–", "-")   # guiones tipográficos → ASCII
    s = s.replace(",", ".")                       # coma decimal → punto
    return s


def parse_number_or_pair(text: str) -> tuple[float | None, float | None]:
    """
    Parsea un string de entrada numérica, con o sin incertidumbre.

    Formatos aceptados:
        "4.340148"
        "4,340148"                   (coma decimal)
        "4,340148 +/- 0,004251"
        "4.340148 ± 0.004251"
        "4.34 +/- 0.01"

    Returns:
        (valor, sigma) si se especificó incertidumbre.
        (valor, None)  si solo se especificó el valor.
        (None,  None)  si el string es inválido o vacío.
    """
    if text is None:
        return None, None
    t = str(text).strip()
    if not t:
        return None, None

    t = t.replace("±", "+/-")

    if "+/-" in t:
        parts = t.split("+/-", 1)
        a = _normalize_num_text(parts[0])
        b = _normalize_num_text(parts[1])
        try:
            return float(a), abs(float(b))
        except Exception:
            return None, None

    t = _normalize_num_text(t)
    try:
        return float(t), None
    except Exception:
        return None, None
