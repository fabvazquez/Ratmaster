# RatMaster v12

Software de planificación dosimétrica para **BNCT** (Boron Neutron Capture Therapy).

---

## Estructura del proyecto

```
ratmaster/
│
├── main.py                  ← Punto de entrada (ejecutar este archivo)
├── app_paths.py             ← Resolución de rutas (desarrollo / PyInstaller)
├── constants.py             ← Constantes globales: órganos, colores, presets
│
├── physics/
│   ├── __init__.py
│   ├── dose_utils.py        ← DVH, métricas dosimétricas, constraints (puras)
│   ├── compute_bnct.py      ← Núcleo de cálculo: dosis física y equivalente
│   └── isoe.py              ← Modelo isoefectivo MLQ (Lea-Catcheside + sinergia)
│
├── data/
│   ├── __init__.py
│   ├── vector_loader.py     ← Carga de .mat con vectores de tasa de dosis
│   └── persistence.py       ← Config JSON, registry SPND, parseo numérico
│
└── ui/
    ├── __init__.py
    ├── formatters.py        ← Formateo de valores para la UI
    ├── canvas.py            ← Widget Matplotlib embebido (MplCanvas)
    ├── splash.py            ← Pantalla de carga inicial
    ├── main_window.py       ← Ventana principal (BNCTMain)
    └── dialogs/
        ├── __init__.py
        ├── results.py       ← ResultsDialog + ComponentDosesDialog
        ├── bio_params.py    ← BioParamsDialog (CBE / RBE)
        ├── boro.py          ← BoroDialog (protocolos de boro)
        ├── isoe_dialogs.py  ← IsoEDialog + IsoEPresetsDialog
        ├── constraints.py   ← ConstraintsDialog + PasteableTable
        ├── spnd.py          ← SPNDConfigDialog + SPNDFromCurrentsDialog
        └── vector_gen.py    ← VectorGenDialog (SEG + Meshtal → .mat)
```

---

## Cómo ejecutar

```bash
# Desde el directorio que contiene la carpeta ratmaster/
python -m ratmaster.main

# O directamente:
python ratmaster/main.py
```

---

## Dependencias

```
PySide6 >= 6.5
numpy
scipy
matplotlib
reportlab      # para exportación PDF (opcional)
```

Instalación:

```bash
pip install PySide6 numpy scipy matplotlib reportlab
```

---

## Guía de módulos

### `physics/` — Núcleo de cálculo (sin Qt)

| Módulo | Qué hace |
|--------|----------|
| `dose_utils.py` | DVH acumulativo exacto, métricas con incertidumbre (Dmax, Dmean, D95, D5, Dmin), resumen de constraints, valor alcanzado por constraint |
| `compute_bnct.py` | Cálculo de dosis física y equivalente por vóxel. Soporta modo tiempo fijo y modo constraints (Dmax, Dmean, Dmin, Dose@Vx%) |
| `isoe.py` | Modelo MLQ con sinergia (González & Santa Cruz 2012). Lea-Catcheside por componente, términos cruzados G_ij, resolución analítica de A(IsoE) |

### `data/` — Persistencia y carga de datos

| Módulo | Qué hace |
|--------|----------|
| `vector_loader.py` | Carga `VectorDoseRate<Organo>.mat` con variables Boro/Fstn/Thn/Gamma. Maneja aliases de nombres (PulmonDerecho → PulmonDer) y búsqueda normalizada |
| `persistence.py` | Registry SPND en JSON, parseo de números con incertidumbre ("4.32 ± 0.01") |

### `ui/` — Interfaz gráfica (PySide6)

| Módulo | Qué hace |
|--------|----------|
| `main_window.py` | Ventana principal. Gestiona DVH con modelo "sin check = todos visibles" y colores fijos por órgano |
| `dialogs/results.py` | Tabla de resultados + DVH. Botón "Componentes de Dosis" abre `ComponentDosesDialog` |
| `dialogs/spnd.py` | Configuración de detectores SPND y cálculo de flujo desde corrientes medidas |
| `dialogs/vector_gen.py` | Generador de vectores desde archivos SEG + Meshtal de MCNP |

---

## Convenciones

- **Órganos**: el orden canónico es `ORG_ORDER` en `constants.py`.
- **Colores DVH**: asignados por índice en `ORGAN_COLORS`; el mismo color se usa en el gráfico y en la lista.
- **Componentes**: Boro, Fstn (neutrones rápidos), Thn (neutrones térmicos), Gamma.
- **Unidades de dosis**: Gy (física) / Gy(RBE) (equivalente) / Gy(IsoE) (isoefectiva).
