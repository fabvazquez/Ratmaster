"""
data/persistence.py
===================
Persistencia de configuración y datos de usuario de RatMaster.

Incluye:
  - DEFAULT_SPND_REGISTRY: detectores SPND con factores de calibración y sensibilidad.
  - load_spnd_registry() / save_spnd_registry(): lectura/escritura del registry JSON.
  - parse_number_or_pair(): parseo robusto de "4.32 ± 0.01" desde entrada de usuario.

Los archivos JSON se guardan en %APPDATA%\\RatMaster\\ (ver app_paths.py).
"""

import json
from ratmaster.app_paths import ensure_user_json


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
    # Fallback: copia del default (deep copy para evitar mutaciones)
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

    # Normalizar símbolo de incertidumbre
    t = t.replace("±", "+/-")

    if "+/-" in t:
        parts = t.split("+/-", 1)
        a = _normalize_num_text(parts[0])
        b = _normalize_num_text(parts[1])
        try:
            return float(a), abs(float(b))
        except Exception:
            return None, None

    # Sin incertidumbre: solo el valor
    t = _normalize_num_text(t)
    try:
        return float(t), None
    except Exception:
        return None, None
