"""
ui/formatters.py
================
Funciones de formateo para mostrar valores numéricos con incertidumbre
y etiquetas de modo de dosis en la interfaz gráfica.
"""


def format_value_uncertainty(value: float, sigma: float, unit: str = "") -> str:
    """
    Formatea un valor con su incertidumbre en notación decimal.
    Ejemplo: 12.34 ± 0.05 Gy

    Usa 2 cifras significativas en sigma para determinar la cantidad de decimales
    del valor. Si sigma es 0 o None, muestra 4 decimales.

    Args:
        value: valor central.
        sigma: incertidumbre (1σ).
        unit:  unidad física (se agrega al final con un espacio).

    Returns:
        String formateado, p.ej. "12.34 ± 0.05 Gy".
    """
    suffix = f" {unit}" if unit else ""
    if not sigma or sigma == 0:
        return f"{value:.4f}{suffix}"
    # Determinar dígitos según el orden de magnitud de sigma
    import math
    try:
        mag = -int(math.floor(math.log10(abs(sigma)))) + 1
        mag = max(0, min(mag, 6))
    except Exception:
        mag = 4
    fmt = f"{{:.{mag}f}}"
    return f"{fmt.format(value)} ± {fmt.format(sigma)}{suffix}"


def format_scientific_value_uncertainty(value: float, sigma: float, unit: str = "") -> str:
    """
    Formatea un valor con incertidumbre en notación científica.
    Ejemplo: 1.23e+09 ± 4.5e+07 n/cm²·s

    Útil para flujos y corrientes con muchos órdenes de magnitud.

    Args:
        value: valor central.
        sigma: incertidumbre (1σ).
        unit:  unidad física.

    Returns:
        String formateado en notación científica.
    """
    suffix = f" {unit}" if unit else ""
    try:
        if sigma and sigma > 0:
            return f"{value:.3e} ± {sigma:.2e}{suffix}"
        return f"{value:.4e}{suffix}"
    except Exception:
        return f"{value}{suffix}"


def dose_type_display(idx: int) -> str:
    """
    Devuelve la etiqueta de modo de dosis según el índice del combo.

    Índices:
        0 → "Física"
        1 → "Equivalente (RBE/CBE)"
        2 → "IsoE (MLQ)"
    """
    return {
        0: "Física",
        1: "Equivalente (RBE/CBE)",
        2: "IsoE (MLQ)",
    }.get(idx, "Física")


def dose_axis_unit(use_bio: bool = False, is_isoe: bool = False) -> str:
    """
    Devuelve la etiqueta de unidad para el eje de dosis según el modo activo.

    Args:
        use_bio:  True si está en modo dosis equivalente (RBE/CBE).
        is_isoe:  True si está en modo IsoE (MLQ).

    Returns:
        String de unidad: "Gy", "Gy(RBE)" o "Gy(IsoE)".
    """
    if is_isoe:
        return "Gy(IsoE)"
    if use_bio:
        return "Gy(RBE)"
    return "Gy"
