"""
app_paths.py
============
Resolución de rutas para RatMaster en modo desarrollo y empaquetado (PyInstaller).

En modo empaquetado (frozen), el ejecutable se encuentra en:
    Program Files\\RatMaster\\RatMaster.exe
    └── Vectores\\          ← datos de vectores, junto al .exe
    └── assets\\            ← íconos y recursos

En modo desarrollo, todas las rutas son relativas al directorio del script.
Los archivos de usuario (config, vectores propios) se escriben en %APPDATA%\\RatMaster.
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Nombre de la aplicación (usado para carpeta de usuario en %APPDATA%)
APP_NAME = "RatMaster"


# ── Detección de modo empaquetado ────────────────────────────────────────────

def is_frozen_app() -> bool:
    """Devuelve True si la app se está ejecutando empaquetada con PyInstaller."""
    return getattr(sys, "frozen", False)


def app_install_dir() -> Path:
    """
    Directorio de instalación de la aplicación.
    - Empaquetado: carpeta que contiene el .exe (donde también está 'Vectores/').
    - Desarrollo:  carpeta que contiene este archivo .py.
    """
    if is_frozen_app():
        exe_dir = Path(sys.executable).parent.resolve()

        # Caso estándar: Vectores está junto al .exe
        if (exe_dir / "Vectores").exists():
            return exe_dir

        # Fallback: dentro de _MEIPASS (bundle onefile)
        if hasattr(sys, "_MEIPASS"):
            meipass = Path(sys._MEIPASS).resolve()
            if (meipass / "Vectores").exists():
                return meipass

        return exe_dir  # último recurso

    # Modo desarrollo: la raíz del proyecto es el PADRE de la carpeta ratmaster/.
    # app_paths.py vive en ratmaster/app_paths.py, así que:
    #   Path(__file__).parent   → ratmaster/
    #   .parent                 → raíz/  (donde están Vectores/, RatMaster.py, bnct_union.py)
    root = Path(__file__).parent.parent.resolve()

    # Doble verificación: si Vectores/ existe en la raíz confirmada, la usamos.
    # Si no (p.ej. alguien copió solo la subcarpeta), devolvemos igual la raíz
    # para que el error de "no se encontró Vectores" sea claro.
    return root


# ── Directorio de usuario (escribible) ───────────────────────────────────────

def user_app_dir() -> Path:
    """
    Carpeta de datos del usuario: %APPDATA%\\RatMaster (Windows).
    Se crea si no existe. Aquí se guardan config, vectores propios y registry SPND.
    """
    base = Path(os.getenv("APPDATA", str(Path.home())))
    p = (base / APP_NAME).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_writable_path(*parts) -> Path:
    """Construye una ruta dentro de user_app_dir() y crea los directorios intermedios."""
    p = user_app_dir().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Recursos empaquetados (solo lectura) ─────────────────────────────────────

def bundled_resource_path(*parts) -> Path:
    """Ruta a un recurso que viene con la instalación (assets, config por defecto, etc.)."""
    return app_install_dir().joinpath(*parts)


def bundled_icon_path() -> Path | None:
    """
    Busca el ícono de la aplicación en las ubicaciones posibles.
    Devuelve la primera que exista, o None si no se encuentra ninguna.
    """
    candidates = [
        bundled_resource_path("assets", "Ratmaster_logo.ico"),
        bundled_resource_path("assets", "Ratmaster_logo.png"),
        bundled_resource_path("Ratmaster_logo.png"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None


# ── Archivos JSON de usuario (config, registry) ───────────────────────────────

def ensure_user_json(
    filename: str,
    default_data: dict | list | None = None,
    bundled_subdir: str = "config",
) -> Path:
    """
    Garantiza que exista un archivo JSON en el directorio del usuario.

    Orden de resolución:
      1. Si ya existe en user_app_dir() → lo devuelve tal cual.
      2. Si existe una copia bundled (en bundled_subdir/ o raíz) → la copia al user dir.
      3. Si se proporcionó default_data → crea el archivo con ese contenido.

    Devuelve siempre la ruta en user_app_dir().
    """
    dst = user_writable_path(filename)
    if dst.exists():
        return dst

    # Buscar copia bundled para copiar como plantilla inicial
    candidates = []
    if bundled_subdir:
        candidates.append(bundled_resource_path(bundled_subdir, filename))
    candidates.append(bundled_resource_path(filename))

    for src in candidates:
        try:
            if src.exists():
                shutil.copy2(src, dst)
                return dst
        except Exception:
            pass

    # Crear con datos por defecto si se proporcionaron
    if default_data is not None:
        try:
            dst.write_text(
                json.dumps(default_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    return dst


# ── Vectores de dosis ─────────────────────────────────────────────────────────

def ensure_user_vectors():
    """
    Copia el directorio 'Vectores/' desde la instalación al directorio del usuario,
    si aún no existe allí. Usa dirs_exist_ok=True para no sobreescribir cambios del usuario.
    """
    src = app_install_dir() / "Vectores"
    dst = user_app_dir() / "Vectores"

    if not src.exists():
        print(f"[RatMaster] Advertencia: no se encontró 'Vectores' en instalación: {src}")
        return

    try:
        shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=shutil.copy2)
    except Exception as e:
        print(f"[RatMaster] Error copiando vectores: {e}")
