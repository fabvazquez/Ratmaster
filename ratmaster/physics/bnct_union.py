from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy.interpolate import RegularGridInterpolator
from scipy.io import savemat
import matplotlib.pyplot as plt

# ===========================
# SEG: lectura + LUT
# ===========================
def load_seg_with_lut(seg_path: str, pixel_size_mm=(1.0, 1.0, 1.0)) -> dict:
    """
    Carga archivo .seg y construye:
      - MaterialNames (lista)
      - Matrix (X,Y,Z) int32
      - LUT: label_real -> nombre_material
      - VoxelSize_mm y Extent_mm
    """
    with open(seg_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    if raw.size < 4:
        raise ValueError("Archivo demasiado corto.")

    X, Y, Z = map(int, raw[:3])
    X = 256 if X == 255 else X
    Y = 256 if Y == 255 else Y
    Z = 256 if Z == 255 else Z
    Mat = int(raw[3])

    p = 4
    materiales = []
    for _ in range(Mat):
        if p >= raw.size:
            raise ValueError("Cabecera truncada.")
        L = int(raw[p]); p += 1
        # a veces hay un 0x00 separador
        if p < raw.size and raw[p] == 0x00:
            p += 1
        if L == 0:
            materiales.append("")
            continue
        if p + L > raw.size:
            raise ValueError("Nombre fuera de rango.")
        name_bytes = raw[p:p+L]
        p += L
        try:
            name = name_bytes.tobytes().decode("utf-8")
        except UnicodeDecodeError:
            name = name_bytes.tobytes().decode("latin-1", errors="replace")
        materiales.append(name)

    expected = X * Y * Z
    remaining = raw.size - p
    if remaining != expected:
        raise ValueError(f"Datos inconsistentes: esperado {expected}, encontrado {remaining}")

    data = raw[p:p+expected]
    matrix = data.reshape((X, Y, Z)).astype(np.int32, copy=False)

    unique_labels = sorted(np.unique(matrix))
    labels_no_zero = [lab for lab in unique_labels if lab != 0]

    if len(materiales) == 0:
        raise ValueError("No se encontraron materiales.")

    lut = {0: materiales[0]}  # 0=fondo (asumido)
    for i, lab in enumerate(labels_no_zero, start=1):
        if i < len(materiales):
            lut[int(lab)] = materiales[i]
        else:
            lut[int(lab)] = f"Org{int(lab)}"

    sx, sy, sz = pixel_size_mm
    extent = (X * sx, Y * sy, Z * sz)

    return {
        "MaterialNames": materiales,
        "Matrix": matrix,
        "LUT": lut,
        "VoxelSize_mm": (sx, sy, sz),
        "Extent_mm": extent
    }


def load_tif_with_lut(tif_path: str,
                      material_names: list[str] | None = None,
                      pixel_size_mm: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Carga un TIFF de segmentación (uint8, 3D) y construye la misma
    estructura que load_seg_with_lut.

    El TIFF no tiene metadatos de materiales, así que los nombres se
    proveen externamente o se generan automáticamente.

    Args:
        tif_path:        path al archivo .tif / .tiff
        material_names:  lista [fondo, mat1, mat2, ...] en el mismo orden
                         que load_seg_with_lut. Si es None se usan nombres
                         genéricos "Org0", "Org1", etc.
                         Tip: pasá seg_result["MaterialNames"] si ya tenés
                         el .seg de referencia.
        pixel_size_mm:   (sx, sy, sz) tamaño de vóxel en mm.

    Retorna el mismo dict que load_seg_with_lut:
        {MaterialNames, Matrix (X,Y,Z) int32, LUT, VoxelSize_mm, Extent_mm}
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "tifffile no está instalado. Instalalo con:  pip install tifffile"
        ) from exc

    with tifffile.TiffFile(tif_path) as tif:
        data = tif.asarray()

    if data.ndim != 3:
        raise ValueError(
            f"Se esperaba un array 3D (Z, Y, X) o (X, Y, Z), "
            f"se obtuvo shape={data.shape}. "
            f"Si es una carpeta de slices 2D, pasalos como lista."
        )

    # tifffile devuelve (Z, Y, X) en la mayoría de los exportadores —
    # transponemos a (X, Y, Z) para coincidir con el formato .seg.
    # Si el shape es cuadrado (256,256,256) no importa, pero lo dejamos
    # explícito para casos no-cúbicos.
    #
    # pixel_size_mm=(sx, sy, sz) sigue la misma convención que en el resto
    # del pipeline y en la UI: sx es el tamaño del vóxel en el eje X de la
    # mesh (eje 0 del matrix resultante), sy en Y (eje 1) y sz en Z (eje 2).
    # La transpose(2,1,0) reordena los datos del TIFF pero NO altera qué
    # tamaño físico le corresponde a cada eje lógico del matrix: eso lo
    # determina el usuario al ingresar los valores en la UI, y se aplican
    # las correcciones geométricas (swap/flip/rot90) a continuación mediante
    # apply_seg_transforms_with_voxelsize, igual que para archivos .seg.
    sx_mat, sy_mat, sz_mat = pixel_size_mm
    matrix = data.transpose(2, 1, 0).astype(np.int32, copy=False)
    X, Y, Z = matrix.shape

    unique_labels = sorted(np.unique(matrix))
    labels_no_zero = [lab for lab in unique_labels if lab != 0]

    # Construir la lista de nombres: el índice 0 es siempre el fondo
    if material_names is None:
        n_total = 1 + len(labels_no_zero)
        material_names = [f"Org{i}" for i in range(n_total)]

    lut: dict[int, str] = {0: material_names[0] if material_names else "Fondo"}
    for i, lab in enumerate(labels_no_zero, start=1):
        if i < len(material_names):
            lut[int(lab)] = material_names[i]
        else:
            lut[int(lab)] = f"Org{int(lab)}"

    extent = (X * sx_mat, Y * sy_mat, Z * sz_mat)

    return {
        "MaterialNames": list(material_names),
        "Matrix":        matrix,
        "LUT":           lut,
        "VoxelSize_mm":  (sx_mat, sy_mat, sz_mat),
        "Extent_mm":     extent,
    }


def load_seg_auto(path: str,
                  pixel_size_mm: tuple = (1.0, 1.0, 1.0),
                  material_names: list[str] | None = None) -> dict:
    """
    Wrapper que elige automáticamente load_seg_with_lut o load_tif_with_lut
    según la extensión del archivo.

    Para .tif/.tiff podés pasar material_names si los tenés; si no, se
    generan nombres genéricos.
    """
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        return load_tif_with_lut(path, material_names=material_names,
                                 pixel_size_mm=pixel_size_mm)
    else:
        # .seg y cualquier otro formato binario propio
        return load_seg_with_lut(path, pixel_size_mm=pixel_size_mm)


# ===========================
# SEG: correcciones geométricas
# ===========================
def swap_YZ(segM: np.ndarray) -> np.ndarray:
    """Intercambia ejes Y y Z: (X,Y,Z) -> (X,Z,Y)"""
    return np.swapaxes(segM, 1, 2)

def rotate_segmentation(segM: np.ndarray) -> np.ndarray:
    """Corrección por defecto (compatibilidad): flip X y flip Z."""
    return np.flip(np.flip(segM, axis=0), axis=2)

# Mapa de nombres de eje a índice
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

# Mapa de nombres de swap a pares de ejes
_SWAP_AXES = {
    "swap_XY": (0, 1),
    "swap_XZ": (0, 2),
    "swap_YZ": (1, 2),
}

def apply_seg_transforms(segM: np.ndarray,
                          transforms: list[dict]) -> tuple[np.ndarray, tuple]:
    """
    Aplica una secuencia arbitraria de transformaciones geométricas a la
    segmentación y calcula el nuevo voxel_size_mm resultante.

    Cada transformación es un dict con la clave "op":
        {"op": "flip_X"}              → np.flip(segM, axis=0)
        {"op": "flip_Y"}              → np.flip(segM, axis=1)
        {"op": "flip_Z"}              → np.flip(segM, axis=2)
        {"op": "swap_XY"}             → np.swapaxes(segM, 0, 1)
        {"op": "swap_XZ"}             → np.swapaxes(segM, 0, 2)
        {"op": "swap_YZ"}             → np.swapaxes(segM, 1, 2)
        {"op": "rot90_XY", "k": 1}    → np.rot90(segM, k=k, axes=(0,1))
        {"op": "rot90_XZ", "k": 1}    → np.rot90(segM, k=k, axes=(0,2))
        {"op": "rot90_YZ", "k": 1}    → np.rot90(segM, k=k, axes=(1,2))

    Retorna:
        (segM_transformed, voxel_size_mm_new)
        voxel_size_mm_new refleja swaps de ejes; las demás ops no cambian el tamaño.

    Nota: voxel_size_mm_new es solo informativo para el swap; rot90 en planos
    con voxels isótropos es transparente; en anisótropos debe verificarse
    manualmente que el resultado físico sea correcto.
    """
    # Mantenemos un arreglo de los tamaños de voxel para trackear swaps
    # (no se pasa voxel_size_mm aquí porque la función es solo de array;
    # el diálogo se encarga de actualizar Seg["VoxelSize_mm"] por separado)
    result = segM.copy() if not segM.flags["C_CONTIGUOUS"] else segM

    for t in transforms:
        op = t.get("op", "")
        if op == "flip_X":
            result = np.flip(result, axis=0)
        elif op == "flip_Y":
            result = np.flip(result, axis=1)
        elif op == "flip_Z":
            result = np.flip(result, axis=2)
        elif op in _SWAP_AXES:
            a0, a1 = _SWAP_AXES[op]
            result = np.swapaxes(result, a0, a1)
        elif op in ("rot90_XY", "rot90_XZ", "rot90_YZ"):
            plane = op.split("_")[1]   # "XY", "XZ" o "YZ"
            ax0 = _AXIS_INDEX[plane[0]]
            ax1 = _AXIS_INDEX[plane[1]]
            k = int(t.get("k", 1))
            result = np.rot90(result, k=k, axes=(ax0, ax1))
        else:
            raise ValueError(f"Transformación desconocida: '{op}'")

    # Calcular voxel_size_mm_new a partir de los swaps (los flips no la cambian)
    # Devolvemos None para que el caller la compute si necesita; es responsabilidad
    # del diálogo actualizar Seg["VoxelSize_mm"] según los swaps que aplicó.
    return result


def apply_seg_transforms_with_voxelsize(segM: np.ndarray,
                                         voxel_size_mm: tuple,
                                         transforms: list[dict]) -> tuple[np.ndarray, tuple]:
    """
    Igual que apply_seg_transforms pero también propaga voxel_size_mm
    a través de los swaps de eje.

    Retorna (segM_transformed, voxel_size_mm_new).
    """
    vs = list(voxel_size_mm)   # [sx, sy, sz]
    result = segM

    for t in transforms:
        op = t.get("op", "")
        if op == "flip_X":
            result = np.flip(result, axis=0)
        elif op == "flip_Y":
            result = np.flip(result, axis=1)
        elif op == "flip_Z":
            result = np.flip(result, axis=2)
        elif op in _SWAP_AXES:
            a0, a1 = _SWAP_AXES[op]
            result = np.swapaxes(result, a0, a1)
            vs[a0], vs[a1] = vs[a1], vs[a0]   # propagar swap al voxel size
        elif op in ("rot90_XY", "rot90_XZ", "rot90_YZ"):
            plane = op.split("_")[1]
            ax0 = _AXIS_INDEX[plane[0]]
            ax1 = _AXIS_INDEX[plane[1]]
            k = int(t.get("k", 1))
            result = np.rot90(result, k=k, axes=(ax0, ax1))
            # rot90 en k impar intercambia las dims de los dos ejes del plano
            if k % 2 == 1:
                vs[ax0], vs[ax1] = vs[ax1], vs[ax0]
        else:
            raise ValueError(f"Transformación desconocida: '{op}'")

    return result, tuple(vs)


# Preset por defecto (compatibilidad con el comportamiento anterior)
DEFAULT_TRANSFORMS = [
    {"op": "swap_YZ"},
    {"op": "flip_X"},
    {"op": "flip_Z"},
]


# ===========================
# Centros físicos eficientes (solo voxeles del órgano)
# ===========================
def organ_voxel_centers_from_indices(indices_ijk: np.ndarray,
                                     voxel_size_mm,
                                     origin_mesh_cm):
    """
    indices_ijk: (N,3) con índices (i,j,k) en la segmentación ya orientada
    Devuelve pts (N,3) en cm con centros de voxel.
    """
    sx_mm, sy_mm, sz_mm = voxel_size_mm
    sx, sy, sz = np.array([sx_mm, sy_mm, sz_mm], dtype=np.float64) / 10.0  # mm -> cm

    i = indices_ijk[:, 0].astype(np.float64)
    j = indices_ijk[:, 1].astype(np.float64)
    k = indices_ijk[:, 2].astype(np.float64)

    x = i * sx + sx / 2 + origin_mesh_cm[0]
    y = j * sy + sy / 2 + origin_mesh_cm[1]
    z = k * sz + sz / 2 + origin_mesh_cm[2]
    return np.column_stack([x, y, z])


# ===========================
# MESHTAL: parse robusto
# ===========================
_num_re = re.compile(r'^[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?$')

def _is_number_token(tok: str) -> bool:
    return _num_re.match(tok) is not None

def _parse_numbers_from_line(line: str):
    toks = line.strip().split()
    return [float(t) for t in toks if _is_number_token(t)]

def _find_all_mesh_blocks(lines):
    starts = []
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("mesh tally number"):
            starts.append(i)
    blocks = []
    for idx, s in enumerate(starts):
        e = starts[idx+1] if idx+1 < len(starts) else len(lines)
        blocks.append((s, e))
    return blocks

def _read_boundaries(lines, start, end, label: str):
    lab = f"{label} direction:".lower()
    for i in range(start, end):
        lo = lines[i].lower()
        if lab in lo:
            acc = []
            j = i
            while j < end and (":" in lines[j] or len(acc) == 0):
                part = lines[j].split(":", 1)[-1] if ":" in lines[j] else lines[j]
                acc.extend(_parse_numbers_from_line(part))
                if j + 1 < end:
                    nxt = lines[j+1].lower()
                    if ("direction:" in nxt) or ("energy bin boundaries" in nxt) or (("result" in nxt) and ("error" in nxt)):
                        break
                j += 1
            return np.array(acc, dtype=float) if acc else None
    return None

def _find_table_header(lines, start, end):
    for i in range(start, end):
        lo = lines[i].lower()
        if ('result' in lo and 'error' in lo) and ('x' in lo and 'y' in lo and 'z' in lo):
            return i
    return None

def _read_table(lines, hdr_idx, end, has_energy_col: bool):
    rows = []
    for ln in lines[hdr_idx+1:end]:
        toks = ln.split()
        if not toks:
            if rows:
                break
            continue
        # Sin energy: X Y Z RESULT ERROR → 5 columnas
        # Con energy: ENERGY X Y Z RESULT ERROR → 6 columnas
        need = 6 if has_energy_col else 5
        if len(toks) < need:
            if rows:
                break
            continue
        cand = toks[:need]
        if all(_is_number_token(t) for t in cand):
            rows.append([float(x) for x in cand])
        else:
            if rows:
                break
    if not rows:
        raise ValueError("No se encontraron filas de datos debajo del encabezado.")
    return np.array(rows, dtype=float)

def _centers_from_boundaries(b: np.ndarray):
    if b is None or b.size < 2:
        return None
    return (b[:-1] + b[1:]) * 0.5

def _reshape_block(vals: np.ndarray, nx: int, ny: int, nz: int):
    if vals.size != nx * ny * nz:
        raise ValueError(f"El tamaño {vals.size} no coincide con nx*ny*nz={nx*ny*nz}.")
    return vals.reshape((nx, ny, nz))

def read_meshtal_all(filename: str) -> dict:
    """
    Lee todas las meshes de un archivo meshtal/meshtally (texto).

    Estructura de retorno por tally:
        meshes[tally_number] = {
            "tally_number": int,
            "centers":      {"x": array, "y": array, "z": array},
            "Matrix":       float32 array (nx, ny, nz) — valores de tally,
            "ErrorMatrix":  float32 array (nx, ny, nz) — errores relativos MCNP.
        }

    Formato del meshtal (columnas):
        Sin energy: X  Y  Z  RESULT  REL_ERROR          (5 columnas)
        Con energy: ENERGY  X  Y  Z  RESULT  REL_ERROR  (6 columnas)

    La columna REL_ERROR es el error relativo estadístico de MCNP (fracción, no %),
    tal como aparece en el archivo: ej. 7.73532E-02 = 7.7%.
    """
    with open(filename, "rt", encoding="latin-1", errors="ignore") as f:
        lines = f.readlines()

    blocks = _find_all_mesh_blocks(lines)
    meshes = {}

    for (b0, b1) in blocks:
        m = re.search(r"mesh tally number\s+(\d+)", lines[b0], flags=re.I)
        tally_number = int(m.group(1)) if m else None

        bx = _read_boundaries(lines, b0, b1, "X")
        by = _read_boundaries(lines, b0, b1, "Y")
        bz = _read_boundaries(lines, b0, b1, "Z")

        cx = _centers_from_boundaries(bx)
        cy = _centers_from_boundaries(by)
        cz = _centers_from_boundaries(bz)

        nx = len(cx) if cx is not None else None
        ny = len(cy) if cy is not None else None
        nz = len(cz) if cz is not None else None

        hdr = _find_table_header(lines, b0, b1)
        if hdr is None:
            print(f"[WARN] No se encontró encabezado de tabla en la malla #{tally_number}.")
            continue

        has_energy_col = ('energy' in lines[hdr].lower().split())
        A = _read_table(lines, hdr, b1, has_energy_col)

        if has_energy_col:
            # Columnas: ENERGY  X  Y  Z  RESULT  REL_ERROR
            #   índices:   0    1  2  3     4        5
            vals = A[:, 4]
            errs = A[:, 5]
            if None in (nx, ny, nz):
                x_u = np.unique(A[:, 1]); y_u = np.unique(A[:, 2]); z_u = np.unique(A[:, 3])
                cx, cy, cz = x_u, y_u, z_u
                nx, ny, nz = x_u.size, y_u.size, z_u.size
        else:
            # Columnas: X  Y  Z  RESULT  REL_ERROR
            #   índices: 0  1  2     3       4
            vals = A[:, 3]
            errs = A[:, 4]
            if None in (nx, ny, nz):
                x_u = np.unique(A[:, 0]); y_u = np.unique(A[:, 1]); z_u = np.unique(A[:, 2])
                cx, cy, cz = x_u, y_u, z_u
                nx, ny, nz = x_u.size, y_u.size, z_u.size

        M = _reshape_block(vals, nx, ny, nz)
        E = _reshape_block(errs, nx, ny, nz)

        meshes[tally_number] = {
            "tally_number": tally_number,
            "centers": {
                "x": np.array(cx, dtype=float),
                "y": np.array(cy, dtype=float),
                "z": np.array(cz, dtype=float),
            },
            "Matrix":      M.astype(np.float32),
            "ErrorMatrix": E.astype(np.float32),
        }
        print(f"Leída malla #{tally_number}: shape {M.shape}, "
              f"Rel Error promedio = {np.nanmean(E):.3f}")

    return meshes


# ===========================
# Helpers internos de orientación de ejes
# ===========================
def _orient_axes(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray,
                 *matrices: np.ndarray):
    """
    Garantiza que los tres ejes estén en orden creciente.
    Aplica los mismos flips a TODAS las matrices pasadas como argumentos.

    Retorna (cx, cy, cz, mat0, mat1, ...)
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    cz = np.asarray(cz, dtype=float)

    mats = [np.asarray(m, dtype=float) for m in matrices]

    if cx.size >= 2 and cx[0] > cx[-1]:
        cx = cx[::-1]
        mats = [np.flip(m, axis=0) for m in mats]

    if cy.size >= 2 and cy[0] > cy[-1]:
        cy = cy[::-1]
        mats = [np.flip(m, axis=1) for m in mats]

    if cz.size >= 2 and cz[0] > cz[-1]:
        cz = cz[::-1]
        mats = [np.flip(m, axis=2) for m in mats]

    return (cx, cy, cz, *mats)

# Mantener la función original por compatibilidad con código existente
def _ensure_increasing_axis(axis_vals: np.ndarray, M: np.ndarray, axis_index: int):
    axis_vals = np.asarray(axis_vals, dtype=float)
    if axis_vals.size < 2:
        return axis_vals, M
    if axis_vals[0] <= axis_vals[-1]:
        return axis_vals, M
    return axis_vals[::-1], np.flip(M, axis=axis_index)


# ===========================
# Interpolación trilineal con manejo correcto de dominio
# ===========================
def calcular_dosis_por_organo_trilineal(segM: np.ndarray,
                                        voxel_size_mm,
                                        origin_mesh_cm,
                                        meshes: dict,
                                        lut: dict,
                                        factor_spnd: float):
    """
    Calcula dosis e incertidumbre MCNP por vóxel para cada órgano de la segmentación.

    Estrategia de dominio:
      - Se calcula una máscara de validez: un vóxel es válido si y solo si
        su centro físico está dentro del dominio de interpolación de TODOS los tallies.
      - Los vóxeles fuera del dominio de cualquier tally se descartan completamente
        (no se extrapola, no se clampea, no se asignan NaN).
      - El vector resultante contiene solo los vóxeles válidos. Se guarda la máscara
        y los índices originales para poder reconstruir el mapa 3D si se necesita.

    Retorna organ_dose con estructura por órgano:
        organ_dose[organ_name] = {
            "valid_mask":          bool array (N_vox_total,) — vóxeles con datos válidos
            "original_indices":    (N_vox_total, 3) — índices ijk en segmentación
            tally_number (int):    float array (N_valid,) — tasas de dosis interpoladas
            "{tally}_err" (str):   float array (N_valid,) — errores relativos MCNP interpolados
        }

    Args:
        segM:           segmentación (X,Y,Z) con labels de órganos
        voxel_size_mm:  (sx, sy, sz) tamaño de vóxel en mm
        origin_mesh_cm: (ox, oy, oz) origen de la segmentación en coordenadas MCNP (cm)
        meshes:         dict de read_meshtal_all — debe tener 'ErrorMatrix'
        lut:            dict label -> nombre de órgano
        factor_spnd:    factor de conversión de unidades (ver comentario en el código)

    Notas sobre factor_spnd:
        dosis = tally_val / factor_spnd
        Verificar unidades antes de usar para evitar errores de escala.
        Ejemplo: si tally en [Gy·cm²/partícula] y factor_spnd en [n/(cm²·s)],
                 resultado en [Gy/s por unidad de flujo de referencia].
    """
    labels = np.unique(segM)
    labels = labels[labels > 0]

    organ_dose = {}

    for label in labels:
        organ_name = lut.get(int(label), f"Org_{int(label)}")
        idx = np.argwhere(segM == label)          # (N_total, 3) índices ijk
        pts = organ_voxel_centers_from_indices(idx, voxel_size_mm, origin_mesh_cm)
        N_total = len(pts)

        # ── Calcular máscara de validez: intersección de todos los dominios ──
        # Un vóxel es válido si cae dentro del rango [min, max] de CADA tally.
        # Se usa una tolerancia flotante mínima (eps_fp) para evitar falsas
        # exclusiones por errores de redondeo en el límite exacto del dominio.
        eps_fp = 1e-10

        valid_mask = np.ones(N_total, dtype=bool)

        # Pre-calcular los ejes ya orientados de cada mesh para reusar abajo
        oriented_meshes: dict = {}

        for tally, mesh in meshes.items():
            cx = mesh["centers"]["x"]
            cy = mesh["centers"]["y"]
            cz = mesh["centers"]["z"]
            M  = mesh["Matrix"]
            E  = mesh.get("ErrorMatrix")

            # Orientar ejes + matrices juntos para garantizar flips consistentes
            if E is not None:
                cx, cy, cz, M, E = _orient_axes(cx, cy, cz, M, E)
            else:
                cx, cy, cz, M = _orient_axes(cx, cy, cz, M)
                E = None

            oriented_meshes[tally] = (cx, cy, cz, M, E)

            # Verificar que el punto está dentro del dominio convexo de la mesh
            in_x = (pts[:, 0] >= cx.min() - eps_fp) & (pts[:, 0] <= cx.max() + eps_fp)
            in_y = (pts[:, 1] >= cy.min() - eps_fp) & (pts[:, 1] <= cy.max() + eps_fp)
            in_z = (pts[:, 2] >= cz.min() - eps_fp) & (pts[:, 2] <= cz.max() + eps_fp)
            valid_mask &= in_x & in_y & in_z

        n_valid   = int(valid_mask.sum())
        n_dropped = N_total - n_valid

        if n_valid == 0:
            print(f"[WARN] '{organ_name}': 0/{N_total} vóxeles dentro del dominio "
                  f"de la mesh. Órgano omitido.")
            continue

        if n_dropped > 0:
            pct_drop = 100.0 * n_dropped / N_total
            print(f"[INFO] '{organ_name}': {n_dropped}/{N_total} vóxeles fuera del "
                  f"dominio de la mesh ({pct_drop:.1f}%) → descartados.")

        pts_valid = pts[valid_mask]   # (N_valid, 3)

        organ_dose[organ_name] = {
            "valid_mask":       valid_mask,
            "original_indices": idx,
        }

        # ── Interpolar valor y error para cada tally ─────────────────────────
        for tally, (cx, cy, cz, M, E) in oriented_meshes.items():

            # bounds_error=True: si el filtrado es correcto nunca debería saltar.
            # Si salta, es un bug en la máscara (mejor que silenciosamente extrapolar).
            interp_val = RegularGridInterpolator(
                (cx, cy, cz), M,
                method="linear",
                bounds_error=True,
            )

            tally_vals = interp_val(pts_valid).astype(np.float64)

            # Aplicar factor de conversión de unidades (ver docstring)
            dosis = np.clip(tally_vals / float(factor_spnd), 0.0, None)
            organ_dose[organ_name][int(tally)] = dosis

            # Interpolar el error relativo MCNP si está disponible
            # El Rel Error se interpola directamente (aproximación válida cuando
            # el campo varía suavemente; conservadora en regiones de alto gradiente).
            if E is not None:
                interp_err = RegularGridInterpolator(
                    (cx, cy, cz), E,
                    method="linear",
                    bounds_error=True,
                )
                err_vals = np.clip(interp_err(pts_valid).astype(np.float64), 0.0, None)
                organ_dose[organ_name][f"{int(tally)}_err"] = err_vals

        print(f"Calculada dosis para '{organ_name}' (label={int(label)}): "
              f"{n_valid}/{N_total} vóxeles válidos")

    return organ_dose


# ===========================
# Bridge: construir vectors y vectors_err para compute_bnct
# ===========================
def build_vectors_and_errors(organ_dose: dict,
                              tally_map: dict | None = None
                              ) -> tuple[dict, dict]:
    """
    Convierte la salida de calcular_dosis_por_organo_trilineal al formato
    que espera compute_bnct().

    Args:
        organ_dose: dict retornado por calcular_dosis_por_organo_trilineal
        tally_map:  dict {tally_number: componente}
                    Componentes válidos: "Boro", "Fstn", "Thn", "Gamma"
                    Default: {14: "Boro", 24: "Thn", 34: "Fstn", 44: "Gamma"}

    Returns:
        vectors:     {organ: (Br_array, Fr_array, Tr_array, Gr_array)}
                     tasas de dosis por vóxel [unidades de organ_dose]
        vectors_err: {organ: (Br_err, Fr_err, Tr_err, Gr_err)}
                     errores relativos MCNP por vóxel (fracción, 0–1)
                     Si no hay ErrorMatrix para un tally, ese array es ceros.
    """
    if tally_map is None:
        tally_map = {14: "Boro", 24: "Thn", 34: "Fstn", 44: "Gamma"}

    # Mapeo inverso: componente -> número de tally
    comp_to_tally = {v: k for k, v in tally_map.items()}

    vectors:     dict = {}
    vectors_err: dict = {}

    for organ, data in organ_dose.items():
        # Determinar tamaño de los vectores válidos desde el primer tally disponible
        n_valid = None
        for t in tally_map:
            if t in data:
                n_valid = len(data[t])
                break

        if n_valid is None:
            print(f"[WARN] '{organ}': no se encontró ningún tally del tally_map. Omitido.")
            continue

        _zero = np.zeros(n_valid, dtype=np.float64)

        arrs: dict[str, np.ndarray] = {}
        errs: dict[str, np.ndarray] = {}

        for comp in ("Boro", "Fstn", "Thn", "Gamma"):
            t = comp_to_tally.get(comp)
            arrs[comp] = np.asarray(data.get(t,          _zero), dtype=np.float64)
            errs[comp] = np.asarray(data.get(f"{t}_err", _zero), dtype=np.float64)

        vectors[organ]     = (arrs["Boro"], arrs["Fstn"], arrs["Thn"], arrs["Gamma"])
        vectors_err[organ] = (errs["Boro"], errs["Fstn"], errs["Thn"], errs["Gamma"])

    return vectors, vectors_err


def resumen_dosis(organ_dose: dict):
    print("\n=== Resumen de dosis por órgano ===")
    for organ, data in organ_dose.items():
        n_total = len(data.get("original_indices", []))
        n_valid = int(data.get("valid_mask", np.array([])).sum())
        print(f"\nÓrgano: {organ}  [{n_valid}/{n_total} vóxeles válidos]")
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.dtype.kind == 'f':
                tag = "err" if str(key).endswith("_err") else "dose"
                print(f"  Tally {key} ({tag}): "
                      f"media = {np.nanmean(val):.3e}, "
                      f"σ = {np.nanstd(val):.3e}, "
                      f"n = {len(val)}")


# ===========================
# Export .mat
# ===========================
def _safe_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_\-]", "", name)
    return name if name else "Org"

def exportar_vectordose_mat(organ_dose: dict,
                            folder_out: str):
    """
    Exporta vectores de dosis a archivos .mat por órgano.
    Solo exporta los vóxeles válidos (ya filtrados).
    Las claves "_err" se exportan como campos separados (ej. Boro_err).
    """
    folder_out = Path(folder_out)
    folder_out.mkdir(parents=True, exist_ok=True)

    tally_map = {14: "Boro", 24: "Thn", 34: "Fstn", 44: "Gamma", 74: "Flujo"}

    resumen_rows = []
    for organ, data in organ_dose.items():
        data_out = {}

        for tally, name in tally_map.items():
            # Valores de dosis
            if tally in data:
                v = np.asarray(data[tally], dtype=float).reshape(-1, 1)
                data_out[name] = v
                resumen_rows.append({
                    "Organ":   organ,
                    "Tipo":    name,
                    "Tally":   tally,
                    "Media":   float(np.nanmean(v)),
                    "DesvStd": float(np.nanstd(v)),
                    "Nvox":    int(v.shape[0])
                })

            # Errores relativos MCNP
            err_key = f"{tally}_err"
            if err_key in data:
                e = np.asarray(data[err_key], dtype=float).reshape(-1, 1)
                data_out[f"{name}_RelErr"] = e

        if data_out:
            organ_file = _safe_name(organ)
            fname = folder_out / f"VectorDoseRate{organ_file}.mat"
            savemat(fname, data_out)
            keys_exported = list(data_out.keys())
            print(f"Guardado {fname.name}: {keys_exported}")
        else:
            print(f"Órgano {organ}: no se encontraron tallies válidos.")

    return resumen_rows


# ===========================
# Helpers de verificación gráfica (coords / slices)
# ===========================
def seg_phys_coords_1d(n: int, voxel_size_mm: float, origin_cm: float) -> np.ndarray:
    """Centros de voxel en cm, 1D."""
    s_cm = float(voxel_size_mm) / 10.0
    return np.arange(n, dtype=np.float64) * s_cm + s_cm / 2 + float(origin_cm)

def mesh_extent_from_centers(c: np.ndarray) -> tuple[float, float]:
    """
    Convierte centros a "edges" aproximados para imshow extent.
    Asume espaciado ~ uniforme.
    """
    c = np.asarray(c, dtype=float)
    if c.size == 1:
        return float(c[0] - 0.5), float(c[0] + 0.5)
    dc = np.diff(c)
    step_left = dc[0]
    step_right = dc[-1]
    left = c[0] - step_left / 2
    right = c[-1] + step_right / 2
    return float(left), float(right)

def safe_colorbar(im, data2d, label=""):
    """
    Agrega colorbar de forma robusta.
    Evita crash si data es constante o todo NaN/Inf.
    """
    arr = np.asarray(data2d, dtype=float)
    finite = np.isfinite(arr)

    if not np.any(finite):
        print(f"[WARN] Colorbar omitida: {label} (todos NaN/Inf)")
        return None

    vmin = np.nanmin(arr[finite])
    vmax = np.nanmax(arr[finite])

    # Rango degenerado => forzar límites
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
        eps = 1.0 if vmin == 0 else abs(vmin) * 0.01 + 1e-12
        im.set_clim(vmin - eps, vmax + eps)
        print(f"[WARN] Rango degenerado en {label}: vmin=vmax={vmin:.3e}. Forzando clim ±{eps:.3e}")

    return plt.colorbar(im, label=label)
