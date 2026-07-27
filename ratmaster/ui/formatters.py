"""
ui/formatters.py
================
Funciones de formateo para mostrar valores numéricos con incertidumbre
y etiquetas de modo de dosis en la interfaz gráfica.

── Regla de redondeo de incertidumbres ───────────────────────────────────────
Se sigue la convención estándar de reporte de resultados experimentales
(la misma que usa PDG/NIST):

  1. La incertidumbre (σ) se redondea a 1 cifra significativa.
  2. EXCEPCIÓN: si esa cifra significativa es un "1" (ej. σ=14, σ=0.18,
     σ=1.9×10⁸), se usan 2 cifras en vez de 1. Motivo: redondear un valor
     que empieza con 1 a una sola cifra introduce un salto relativo de
     hasta ~50% (ej. 14 → 10 es un 29% de cambio; 19 → 20 es un 5%, pero
     19 → 2×10¹ pierde el detalle de si era 15 o 19). Con 2 cifras el
     salto máximo se acota a ~5%, igual que para cualquier otra cifra líder.
  3. El valor central se redondea al mismo decimal en el que quedó la
     cifra significativa de σ (nunca se le inventan más decimales de los
     que la incertidumbre justifica, ni se le sacan decimales que sí hacen
     falta).
  4. Si el orden de magnitud del VALOR CENTRAL (no de σ) es muy chico o
     muy grande (|exponente en base 10| ≥ 3, es decir <0.001 o ≥1000),
     se usa notación científica explícita: "(mantisa_valor ± mantisa_σ)×10ⁿ".
  5. Si σ es 0 (sin incertidumbre asociada), se muestra el valor solo,
     con 2 decimales si está en rango normal, o en notación científica
     si su magnitud lo amerita — sin inventar una precisión que no existe.

Ver round_value_with_uncertainty() para la implementación y
round_to_n_sig_figs() si en algún momento se quiere probar con una
cantidad distinta de cifras significativas (parámetro `sig_figs`).
"""

import math

# Umbral de |exponente en base 10| del VALOR CENTRAL a partir del cual se
# usa notación científica en vez de notación decimal normal.
SCI_NOTATION_EXPONENT_THRESHOLD = 3

_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _to_superscript(n: int) -> str:
    """Convierte un entero a su representación en superíndice unicode (ej. -3 -> ⁻³)."""
    out = ""
    for c in str(n):
        out += "⁻" if c == "-" else _SUPERSCRIPT_DIGITS[int(c)]
    return out


def round_to_n_sig_figs(x: float, n: int = 1) -> tuple[float, int]:
    """
    Redondea `x` a `n` cifras significativas.

    Devuelve (valor_redondeado, exponente_en_base_10_de_la_primera_cifra_sig).
    El exponente se recalcula DESPUÉS de redondear, para manejar el caso
    borde en que el redondeo sube de categoría (ej. 0.96 con n=1 -> 1.0,
    que pasa de exponente -1 a exponente 0).

    n=1 (default) es la cifra significativa "cruda"; para aplicar la
    excepción del PDG (cifra líder=1 -> 2 cifras) usar
    effective_sig_figs_for_uncertainty() para decidir `n` antes de llamar
    a esta función.
    """
    if x == 0:
        return 0.0, 0
    exp = math.floor(math.log10(abs(x)))
    factor = 10 ** (exp - (n - 1))
    rounded = round(x / factor) * factor
    if rounded != 0:
        exp = math.floor(math.log10(abs(rounded)))
    return rounded, exp


def effective_sig_figs_for_uncertainty(sigma: float) -> int:
    """
    Cantidad de cifras significativas a usar para una incertidumbre,
    aplicando la excepción estándar del "1 inicial": si la primera cifra
    significativa de sigma (SIN redondear todavía) es 1, se usan 2 cifras
    en vez de 1. Para cualquier otra cifra líder (2-9), se usa 1.
    """
    if not sigma or sigma == 0:
        return 1
    exp = math.floor(math.log10(abs(sigma)))
    leading_digit = int(abs(sigma) / (10 ** exp))
    return 2 if leading_digit == 1 else 1


def round_value_with_uncertainty(value: float, sigma: float) -> tuple[float, float, int | None]:
    """
    Redondea sigma a su cantidad de cifras significativas efectiva (ver
    effective_sig_figs_for_uncertainty), y el valor central al mismo
    decimal en el que quedó esa cifra de sigma.

    Devuelve (value_redondeado, sigma_redondeada, decimales_a_mostrar).
    `decimales_a_mostrar` es siempre >= 0 (listo para usar en f"{:.Nf}");
    si la cifra significativa de sigma cae en decenas/centenas/etc., el
    redondeo real ya se aplicó sobre value/sigma antes de llegar a este
    punto (quedan como val.0, val.00, etc., sin necesidad de decimales).

    Si sigma es 0 o None, devuelve (value, 0.0, None) — None indica
    "sin incertidumbre", a tratar aparte por el llamador (no hay cifra
    significativa de sigma que dicte el redondeo del valor).
    """
    if not sigma or sigma == 0:
        return value, 0.0, None

    sig_figs = effective_sig_figs_for_uncertainty(sigma)
    sigma_r, exp_sigma = round_to_n_sig_figs(sigma, sig_figs)

    # decimales "exactos" (puede ser negativo si la cifra sig. de sigma
    # cae en decenas/centenas; Python round() soporta decimales negativos)
    decimals_exact = -(exp_sigma - (sig_figs - 1))
    value_r = round(value, decimals_exact)
    sigma_r = round(sigma_r, decimals_exact)  # limpia residuos de punto flotante
    decimals_to_show = max(decimals_exact, 0)
    return value_r, sigma_r, decimals_to_show


def format_value_uncertainty(
    value: float,
    sigma: float,
    unit: str = "",
    sci_threshold: int = SCI_NOTATION_EXPONENT_THRESHOLD,
) -> str:
    """
    Formatea un valor con su incertidumbre, redondeando sigma a su cifra
    significativa efectiva (1, o 2 si la cifra líder es "1") y el valor al
    mismo decimal. Pasa a notación científica explícita
    "(mantisa_valor ± mantisa_σ)×10ⁿ" si |exponente del VALOR central| >= sci_threshold.

    Ejemplos:
        format_value_uncertainty(18.7, 7.2, "ppm")   -> "19 ± 7 ppm"
        format_value_uncertainty(5.532, 0.18, "Gy")  -> "5.53 ± 0.18 Gy"   (excepción del 1)
        format_value_uncertainty(0.00345, 0.000067)  -> "(3.45 ± 0.07)×10⁻³"
        format_value_uncertainty(3.2, 0)              -> "3.20"            (sin incertidumbre)

    Args:
        value: valor central.
        sigma: incertidumbre (1σ). Si es 0/None, se muestra solo el valor.
        unit:  unidad física (se agrega al final con un espacio).
        sci_threshold: a partir de qué |exponente del valor| usar notación científica.

    Returns:
        String formateado.
    """
    suffix = f" {unit}" if unit else ""

    # Caso sin incertidumbre: no hay sigma que dicte el redondeo del valor.
    # Se muestra con 2 decimales (precisión "razonable" sin inventar nada),
    # o en notación científica si la magnitud lo amerita.
    if not sigma or sigma == 0:
        try:
            exp_v = math.floor(math.log10(abs(value))) if value != 0 else 0
        except (ValueError, OverflowError):
            exp_v = 0
        if abs(exp_v) >= sci_threshold:
            mant = value / (10 ** exp_v)
            return f"{mant:.2f}×10{_to_superscript(exp_v)}{suffix}"
        return f"{value:.2f}{suffix}"

    value_r, sigma_r, decimals = round_value_with_uncertainty(value, sigma)

    try:
        exp_v = math.floor(math.log10(abs(value_r))) if value_r != 0 else 0
    except (ValueError, OverflowError):
        exp_v = 0

    if abs(exp_v) >= sci_threshold:
        # Notación científica explícita: (mantisa_valor ± mantisa_σ)×10^exp_v
        # El exponente se decide por la magnitud del VALOR central (no de σ).
        mant_v = value_r / (10 ** exp_v)
        mant_s = sigma_r / (10 ** exp_v)
        sig_figs_mant = effective_sig_figs_for_uncertainty(sigma_r)
        mant_s_r, exp_mant_s = round_to_n_sig_figs(mant_s, sig_figs_mant)
        dec_mant = max(-(exp_mant_s - (sig_figs_mant - 1)), 0)
        return f"({mant_v:.{dec_mant}f} ± {mant_s_r:.{dec_mant}f})×10{_to_superscript(exp_v)}{suffix}"

    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(value_r)} ± {fmt.format(sigma_r)}{suffix}"


def format_scientific_value_uncertainty(value: float, sigma: float, unit: str = "") -> str:
    """
    Formatea un valor con incertidumbre, forzando notación científica
    explícita independientemente de su magnitud. Útil para flujos y
    corrientes que casi siempre están en notación científica por
    convención, aunque su exponente puntualmente caiga cerca de 1.

    Ejemplo: format_scientific_value_uncertainty(7.19e8, 1.9e8, "n/cm²·s")
             -> "(7.2 ± 1.9)×10⁸ n/cm²·s"

    Sigue la misma regla de cifras significativas que format_value_uncertainty
    (1 cifra para σ, o 2 si la cifra líder es "1").

    Args:
        value: valor central.
        sigma: incertidumbre (1σ).
        unit:  unidad física.

    Returns:
        String formateado en notación científica explícita.
    """
    suffix = f" {unit}" if unit else ""
    try:
        if not sigma or sigma == 0:
            exp_v = math.floor(math.log10(abs(value))) if value != 0 else 0
            mant = value / (10 ** exp_v)
            return f"{mant:.2f}×10{_to_superscript(exp_v)}{suffix}"

        value_r, sigma_r, _ = round_value_with_uncertainty(value, sigma)
        exp_v = math.floor(math.log10(abs(value_r))) if value_r != 0 else 0
        mant_v = value_r / (10 ** exp_v)
        mant_s = sigma_r / (10 ** exp_v)
        sig_figs_mant = effective_sig_figs_for_uncertainty(sigma_r)
        mant_s_r, exp_mant_s = round_to_n_sig_figs(mant_s, sig_figs_mant)
        dec_mant = max(-(exp_mant_s - (sig_figs_mant - 1)), 0)
        return f"({mant_v:.{dec_mant}f} ± {mant_s_r:.{dec_mant}f})×10{_to_superscript(exp_v)}{suffix}"
    except (ValueError, OverflowError):
        return f"{value}{suffix}"


def format_value_uncertainty_separate(value: float, sigma: float) -> tuple[str, str]:
    """
    Igual que format_value_uncertainty(), pero devuelve el valor y la
    incertidumbre como dos strings SEPARADOS (sin combinar en "valor ± σ"
    ni envolver en unidad) — para tablas con columnas independientes de
    valor y "±" (ej. la tabla de métricas DVH en ui/dialogs/results.py).

    Si el resultado requeriría notación científica (|exponente del valor|
    >= sci_threshold), ambos strings incluyen el "×10ⁿ" por separado, ya
    que no hay una sola celda combinada donde factorizarlo:
        format_value_uncertainty_separate(2500, 3) -> ("2.500×10³", "0.003×10³")

    Returns:
        (valor_str, sigma_str)
    """
    if not sigma or sigma == 0:
        try:
            exp_v = math.floor(math.log10(abs(value))) if value != 0 else 0
        except (ValueError, OverflowError):
            exp_v = 0
        if abs(exp_v) >= SCI_NOTATION_EXPONENT_THRESHOLD:
            mant = value / (10 ** exp_v)
            return f"{mant:.2f}×10{_to_superscript(exp_v)}", ""
        return f"{value:.2f}", ""

    value_r, sigma_r, decimals = round_value_with_uncertainty(value, sigma)
    try:
        exp_v = math.floor(math.log10(abs(value_r))) if value_r != 0 else 0
    except (ValueError, OverflowError):
        exp_v = 0

    if abs(exp_v) >= SCI_NOTATION_EXPONENT_THRESHOLD:
        mant_v = value_r / (10 ** exp_v)
        mant_s = sigma_r / (10 ** exp_v)
        sig_figs_mant = effective_sig_figs_for_uncertainty(sigma_r)
        mant_s_r, exp_mant_s = round_to_n_sig_figs(mant_s, sig_figs_mant)
        dec_mant = max(-(exp_mant_s - (sig_figs_mant - 1)), 0)
        sup = _to_superscript(exp_v)
        return f"{mant_v:.{dec_mant}f}×10{sup}", f"{mant_s_r:.{dec_mant}f}×10{sup}"

    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value_r), fmt.format(sigma_r)


def format_time(value: float, sigma: float = 0.0, fmt: str = "s") -> str:
    """
    Formatea un tiempo con su incertidumbre, nunca en notación científica.

    A diferencia de format_value_uncertainty(), esta función no pasa a
    notación científica aunque el valor sea grande — un tiempo siempre
    es más legible como "1234 ± 19 s" o "20m 34s ± 19s" que como
    "1.234×10³ s".

    Args:
        value: tiempo en segundos.
        sigma: incertidumbre en segundos (1σ). 0 = sin incertidumbre.
        fmt:   "s"   → muestra en segundos puros ("612 ± 19 s")
               "ms"  → muestra en "Xm Ys ± Zs"  ("10m 12s ± 19s")
                        Si value < 60s, muestra igual que "s".

    Returns:
        String formateado.
    """
    if fmt == "ms" and value >= 60.0:
        mins = int(value) // 60
        secs = value - mins * 60
        if not sigma or sigma == 0:
            return f"{mins}m {secs:.0f}s"
        # La incertidumbre siempre en segundos (es la unidad natural de sigma)
        _, sigma_r, decimals = round_value_with_uncertainty(secs, sigma)
        fmt_s = f"{{:.{decimals}f}}"
        return f"{mins}m {fmt_s.format(secs)}s ± {fmt_s.format(sigma_r)}s"

    # Formato en segundos puros — igual que format_value_uncertainty
    # pero forzando la ruta decimal (sin umbral de notación científica)
    if not sigma or sigma == 0:
        return f"{value:.2f} s"
    value_r, sigma_r, decimals = round_value_with_uncertainty(value, sigma)
    fmt_s = f"{{:.{decimals}f}}"
    return f"{fmt_s.format(value_r)} ± {fmt_s.format(sigma_r)} s"



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


def parse_time_input(text: str, fmt: str = "s") -> float | None:
    """
    Parsea un tiempo ingresado por el usuario, devolviendo el valor en
    segundos (float). Devuelve None si el texto no se puede interpretar.

    Args:
        text: texto ingresado por el usuario.
        fmt:  "s"  → espera un número simple en segundos ("612", "612.4").
              "ms" → espera formato "Xm Ys" o "X Y" (minutos y segundos,
                     separados por 'm'/espacio), ej: "10m 12s", "10m12s",
                     "10 12". También acepta solo segundos si no hay 'm'
                     en el texto (ej. "612"), por tolerancia.

    Returns:
        Tiempo en segundos, o None si no se pudo interpretar.
    """
    txt = str(text).strip().replace(",", ".")
    if txt == "":
        return None

    if fmt == "s":
        try:
            return float(txt)
        except ValueError:
            return None

    # fmt == "ms": "10m 12s", "10m12s", "10 12", o solo "612" (tolerancia)
    txt_clean = txt.lower().replace("s", "").strip()
    if "m" in txt_clean:
        try:
            mins_str, secs_str = txt_clean.split("m", 1)
            mins = float(mins_str.strip()) if mins_str.strip() else 0.0
            secs = float(secs_str.strip()) if secs_str.strip() else 0.0
            return mins * 60.0 + secs
        except ValueError:
            return None
    # Sin 'm': puede ser "10 12" (min seg separados por espacio) o un
    # número simple de segundos. Se prioriza "número simple" si no hay
    # espacios — es la interpretación menos sorprendente.
    parts = txt_clean.split()
    if len(parts) == 2:
        try:
            mins = float(parts[0])
            secs = float(parts[1])
            return mins * 60.0 + secs
        except ValueError:
            return None
    try:
        return float(txt_clean)
    except ValueError:
        return None


def format_boro_origin(meta: dict) -> str:
    """
    Describe de dónde salió la concentración de boro por órgano usada en el
    cálculo, para mostrar en el reporte de parámetros.

    - Modo "genérico" (o ausente, por compatibilidad con reportes viejos que
      no tenían este campo): "Genérico (valores del protocolo)".
    - Modo "sangre": "A partir de sangre (concentración ± error ppm)".

    Args:
        meta: dict de metadatos del reporte (report["meta"]).

    Returns:
        String descriptivo listo para mostrar.
    """
    source = (meta.get("boro_source_mode") or "generico").strip().lower()
    if source != "sangre":
        return "Genérico (valores del protocolo)"

    conc = meta.get("boro_blood_conc")
    err = meta.get("boro_blood_conc_err") or 0.0
    if conc is None:
        return "A partir de sangre"
    return f"A partir de sangre ({format_value_uncertainty(float(conc), float(err), 'ppm')})"


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
