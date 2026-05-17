"""
data/vector_loader.py
=====================
Carga de vectores de tasa de dosis desde archivos .mat de MCNP/FMESH.

Cada archivo tiene la forma:
    VectorDoseRate<Organo>.mat
con variables internas: Boro, Fstn, Thn, Gamma
que son vectores 1D [n_voxels] con tasas de dosis [Gy·cm²/n].

La carga incluye:
  - Resolución de aliases (PulmonDerecho → PulmonDer, etc.)
  - Fallback por comparación normalizada de nombres (sin acentos, case-insensitive)
"""


import re
import unicodedata


def _safe_set_name(name: str) -> str:
    """
    Normaliza el nombre de una subcarpeta de vectores:
    sin espacios, sin tildes, solo caracteres [A-Za-z0-9_-].

    Usado para leer y escribir la clave 'active_vector_set' en la config.
    Ejemplo: "Set Prueba 2024" → "Set_Prueba_2024"
    """
    name = str(name).strip()
    if not name:
        return "DEFAULT"
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name if name else "DEFAULT"


import unicodedata
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# ── Helpers de normalización de nombres ──────────────────────────────────────

def _strip_accents(s: str) -> str:
    """Elimina diacríticos (tildes, diéresis, etc.) usando NFD."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )


def _norm_name(s: str) -> str:
    """
    Normalización agresiva para comparar nombres de órganos o archivos:
    sin acentos, minúsculas, solo caracteres alfanuméricos.
    """
    s = _strip_accents(str(s)).lower()
    return "".join(ch for ch in s if ch.isalnum())


# ── Tabla de aliases de nombres de archivos ───────────────────────────────────

# Algunos generadores exportan los pulmones con nombres distintos.
# La clave es el nombre CANÓNICO que usa RatMaster internamente.
ORG_FILE_ALIASES: dict[str, list[str]] = {
    "PulmonDer": [
        "PulmonDer", "PulmonDerecho", "PulmonDcho",
        "PulmonRight", "Pulmon_Der", "Pulmon_Derecho", "PulmonDerecha",
    ],
    "PulmonIzq": [
        "PulmonIzq", "PulmonIzquierdo", "PulmonIzqdo",
        "PulmonLeft", "Pulmon_Izq", "Pulmon_Izquierdo", "PulmonIzquierda",
    ],
}


# ── Carga de .mat ─────────────────────────────────────────────────────────────

def safe_loadmat_vars(path: str | Path) -> dict:
    """
    Carga un archivo .mat y filtra las variables de metadatos (__xxx__).
    Devuelve un dict limpio {nombre_variable: valor}.
    """
    raw = loadmat(str(path))
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def load_vectordose(folder: str | Path, organ: str) -> tuple | None:
    """
    Carga el archivo VectorDoseRate<organ>.mat con las cuatro componentes
    de tasa de dosis (Boro, Fstn, Thn, Gamma) como arrays 1D float.

    Estrategia de búsqueda:
      1. Nombre canónico + aliases definidos en ORG_FILE_ALIASES.
      2. Fallback: escanear todos los VectorDoseRate*.mat y comparar
         por nombre normalizado (sin acentos, case-insensitive).

    Args:
        folder: directorio que contiene los archivos .mat.
        organ:  nombre canónico del órgano (ej: "Cerebro", "PulmonIzq").

    Returns:
        Tupla (Boro, Fstn, Thn, Gamma) como arrays 1D de float64,
        o None si no se encontró ningún archivo compatible.

    Raises:
        KeyError: si el archivo existe pero le falta alguna variable requerida.
    """
    folder = Path(folder)

    # ── Intento 1: candidatos por nombre canónico y aliases ──────────────────
    candidates = [organ] + ORG_FILE_ALIASES.get(organ, [])

    for cand in candidates:
        fpath = folder / f"VectorDoseRate{cand}.mat"
        if fpath.exists():
            return _load_mat_four_channels(fpath)

    # ── Intento 2: búsqueda fuzzy por normalización ───────────────────────────
    target = _norm_name(organ)
    for fpath in sorted(folder.glob("VectorDoseRate*.mat")):
        stem = fpath.stem[len("VectorDoseRate"):]   # nombre sin el prefijo
        if _norm_name(stem) == target:
            return _load_mat_four_channels(fpath)

    # No se encontró ningún archivo compatible
    return None


def _load_mat_four_channels(fpath: Path) -> tuple[np.ndarray, ...]:
    """
    Carga las cuatro variables (Boro, Fstn, Thn, Gamma) de un .mat.
    Devuelve (B, F, T, G) como arrays 1D float64.

    Raises:
        KeyError: si alguna variable está ausente.
    """
    d = safe_loadmat_vars(fpath)
    required = ("Boro", "Fstn", "Thn", "Gamma")
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(
            f"{fpath.name}: faltan las variables {missing}. "
            f"Variables presentes: {list(d.keys())}"
        )
    B = np.ravel(d["Boro"]).astype(float)
    F = np.ravel(d["Fstn"]).astype(float)
    T = np.ravel(d["Thn"]).astype(float)
    G = np.ravel(d["Gamma"]).astype(float)
    return B, F, T, G
