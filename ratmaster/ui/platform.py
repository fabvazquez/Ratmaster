"""
ui/platform.py
==============
Adaptaciones de la ventana nativa por sistema operativo.

Actualmente solo implementa soporte para Windows (DWM).
En macOS / Linux las funciones son no-ops silenciosos.

Exporta:
    apply_custom_titlebar(hwnd, bg_hex, text_hex) → None
        Colorea la barra de título de Windows usando la API DWM,
        para que coincida con el color de la barra de menú de la app.
"""

import sys


def apply_custom_titlebar(
    hwnd: int,
    bg_hex: str = "#37474F",
    text_hex: str = "#FFFFFF",
) -> None:
    """
    Colorea la barra de título de Windows para que coincida con la barra de menú.

    Usa la API DWM (Desktop Window Manager):
      - Windows 11 (build 22000+): color exacto de fondo y texto vía
        DWMWA_CAPTION_COLOR (35) y DWMWA_TEXT_COLOR (36).
      - Windows 10: fallback a modo oscuro del sistema (barra gris oscura)
        vía DWMWA_USE_IMMERSIVE_DARK_MODE (20).
      - Otros SO: no hace nada.

    Args:
        hwnd:     Handle nativo de la ventana. Obtener con int(win.winId())
                  *después* de llamar a win.show().
        bg_hex:   Color de fondo de la barra (#RRGGBB).
        text_hex: Color del texto de la barra (#RRGGBB).
    """
    if sys.platform != "win32":
        return

    import ctypes

    def _hex_to_colorref(h: str) -> int:
        """Convierte '#RRGGBB' al formato COLORREF de Windows (0x00BBGGRR)."""
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return r | (g << 8) | (b << 16)

    dwm = ctypes.windll.dwmapi

    def _set_attr(attr_id: int, value: int) -> bool:
        try:
            v = ctypes.c_uint32(value)
            return dwm.DwmSetWindowAttribute(
                hwnd, attr_id, ctypes.byref(v), ctypes.sizeof(v)
            ) == 0  # S_OK = 0
        except Exception:
            return False

    bg_ref   = _hex_to_colorref(bg_hex)
    text_ref = _hex_to_colorref(text_hex)

    # Windows 11: color exacto de fondo, texto y borde
    ok = _set_attr(35, bg_ref)    # DWMWA_CAPTION_COLOR
    if ok:
        _set_attr(36, text_ref)   # DWMWA_TEXT_COLOR
        _set_attr(34, bg_ref)     # DWMWA_BORDER_COLOR (elimina el borde blanco)
    else:
        # Windows 10 fallback: activar modo oscuro del sistema
        _set_attr(20, 1)          # DWMWA_USE_IMMERSIVE_DARK_MODE
