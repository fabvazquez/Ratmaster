"""
RatMaster.py  —  Launcher
===============================


Estructura esperada:
    📁 carpeta programa/
    ├── RatMaster.py            
    ├── bnct_union.py        
    └── 📁 ratmaster/        
        ├── main.py
        ├── physics/
        ├── data/
        └── ui/

Por qué existe este archivo:
    Cuando Python corre un script, agrega su carpeta a sys.path.
    Al correr este launcher desde la raíz, la raíz queda en sys.path
    y "import ratmaster" funciona sin configuración adicional.
"""

import sys
from pathlib import Path

# La raíz del proyecto = carpeta de este archivo.
# Agregarlo a sys.path garantiza que "import ratmaster" funcione.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ratmaster.main import main

if __name__ == "__main__":
    main()
