"""Arma notebooks/TP_individual.ipynb a partir de los módulos de src/.

El notebook de la entrega no se escribe a mano: se genera. Así el código vive en
un solo lugar (los .py, que son los que se revisan y diffean) y el .ipynb es un
artefacto reproducible que se regenera al cerrar cada rama.

Uso:  python -m src.armar_notebook
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from src import config

DESTINO = config.RAIZ / "notebooks" / "TP_individual.ipynb"

# Líneas que sólo tienen sentido en el paquete y estorban en un notebook plano.
DESCARTAR = ("from __future__ import annotations", "from src", "import argparse")

# Módulos a volcar, en orden, con el título de su sección. Al agregar un módulo
# nuevo en las próximas ramas, se suma acá y listo.
# El orden importa: un módulo tiene que estar antes que otro si el otro lo usa al
# definirse, no sólo al ejecutarse. `carga.unir` evalúa config.HOW_UNION en el valor
# por defecto de un parámetro, y `correr_limpieza.PASOS` referencia funciones de
# `limpieza` a nivel de módulo.
SECCIONES = [
    ("src/config.py", "1. Configuración"),
    ("src/carga.py", "2. Carga y unión de las tres tablas"),
    ("src/diagnostico.py", "3. Diagnóstico de la unión"),
    ("src/eda.py", "4. Análisis exploratorio"),
    ("src/genero_por_nombre.py", "5. Diccionario de género por nombre"),
    ("src/limpieza.py", "6. Limpieza"),
    ("src/modelo.py", "7. Modelo congelado de evaluación"),
    ("src/experimentos.py", "8. Registro de experimentos"),
    ("src/correr_base.py", "9. Experimento #0 — la base sin limpiar"),
    ("src/correr_limpieza.py", "10. Los experimentos de limpieza"),
]

# Cuatro módulos definen `main()`. En el paquete conviven sin problema porque cada uno
# tiene su espacio de nombres; en un notebook plano el último pisaría a los anteriores
# y quedaría una sola función `main`. El generador les pone el nombre del módulo.
NOMBRE_MAIN = "def main("

PORTADA = """# TP Individual — QuéLibroLeo
### E72.1.01 · Fundamentos de Métodos Analíticos Predictivos

**Problema.** Predecir si a un lector le va a gustar un libro que no leyó, a partir de
los datos de quelibroleo.com. Es una **clasificación binaria**: `rating >= 7` es que le
gustó (`1`), `rating <= 5` que no (`0`), y los `rating == 6` se descartan por ser un
valor gris que no aporta información.

**Métrica de decisión: `f1`.** El dataset está desbalanceado (~84/16), así que `accuracy`
no sirve para decidir nada. Todas las comparaciones entre experimentos usan `f1` y sólo
`f1`; la matriz de confusión se muestra como material descriptivo.

**Semilla: 42** en todo (split, modelos, samplers, muestreos), para que los errores sean
comparables entre corridas.

> Este notebook se genera automáticamente desde los módulos de `src/` con
> `python -m src.armar_notebook`. No editarlo a mano: los cambios se hacen en los `.py`.
"""

DATOS_MD = """## 0. Los datos

Los tres CSV no están en el repositorio. En Colab, montá el Drive y apuntá
`QLL_DATA_DIR` a la carpeta que los contiene. Si corrés local con los CSV al lado del
notebook, no hace falta tocar nada.
"""

DATOS_CODE = '''import os

# --- Colab: descomentar estas dos líneas y ajustar la ruta ---
# from google.colab import drive; drive.mount("/content/drive")
# os.environ["QLL_DATA_DIR"] = "/content/drive/MyDrive/TP_QueLibroLeo/data"

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
'''

ALIAS_MD = """En los `.py` el código está repartido en módulos y se referencia como `config.SEED` o
`carga.unir(...)`. En el notebook es todo un mismo espacio de nombres, así que estos alias
hacen que esas referencias sigan funcionando sin tener que reescribir una sola línea.
"""

ALIAS_CODE = '''import sys

_yo = sys.modules["__main__"]
config = carga = diagnostico = eda = limpieza = modelo = experimentos = _yo
'''

EJECUCION_MD = """## 11. Ejecución

Las tres etapas, en orden. Cada `main` lleva el nombre de su módulo porque en un
notebook todo comparte el mismo espacio de nombres.

`main_correr_limpieza` reconstruye la tabla de experimentos entera, incluida la fila
del #0, así que no hace falta llamar a `main_correr_base` por separado.
"""

EJECUCION_CODE = """%%time
base = main_diagnostico()      # carga, une, construye el target y deja 01_base.pkl
main_eda()                     # las figuras del informe
tabla = main_correr_limpieza() # los experimentos de limpieza y 04_limpio.pkl
"""


def cuerpo_del_modulo(ruta: Path) -> tuple[str, str]:
    """Separa el docstring del módulo del resto del código, ya limpio para el notebook.

    Descarta los imports del paquete y el bloque `if __name__ == "__main__"`, que en un
    notebook no aplican.
    """
    fuente = ruta.read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    lineas = fuente.splitlines()

    docstring = ast.get_docstring(arbol) or ""
    desde = arbol.body[0].end_lineno if docstring else 0

    # El bloque __main__ es el guard de línea de comandos: en el notebook sobra.
    corte = len(lineas)
    for nodo in arbol.body:
        if isinstance(nodo, ast.If) and "__main__" in ast.unparse(nodo.test):
            corte = nodo.lineno - 1
            break

    codigo = [ln for ln in lineas[desde:corte]
              if not any(ln.startswith(p) for p in DESCARTAR)]
    fuente_limpia = "\n".join(codigo).strip() + "\n"

    # `main` se renombra con el nombre del módulo. Es seguro: cada módulo la define
    # una sola vez y la llama únicamente desde el guard `__main__`, que ya se descartó.
    nombre_main = None
    if NOMBRE_MAIN in fuente_limpia:
        nombre_main = f"main_{ruta.stem}"
        fuente_limpia = fuente_limpia.replace(NOMBRE_MAIN, f"def {nombre_main}(")

    return docstring, fuente_limpia, nombre_main


def celda(tipo: str, texto: str) -> dict:
    fuente = texto.splitlines(keepends=True)
    base = {"cell_type": tipo, "metadata": {}, "source": fuente}
    return base if tipo == "markdown" else {**base, "execution_count": None, "outputs": []}


def armar() -> dict:
    celdas = [
        celda("markdown", PORTADA),
        celda("markdown", DATOS_MD),
        celda("code", DATOS_CODE),
    ]

    for i, (ruta, titulo) in enumerate(SECCIONES):
        docstring, codigo, _ = cuerpo_del_modulo(config.RAIZ / ruta)
        celdas.append(celda("markdown", f"## {titulo}\n\n{docstring}\n\n*(desde `{ruta}`)*\n"))
        celdas.append(celda("code", codigo))
        # Los alias van después de config y antes del resto, porque `unir` los usa
        # en el valor por defecto de un parámetro y eso se evalúa al definir la función.
        if i == 0:
            celdas.append(celda("markdown", ALIAS_MD))
            celdas.append(celda("code", ALIAS_CODE))

    celdas += [celda("markdown", EJECUCION_MD), celda("code", EJECUCION_CODE)]

    return {
        "cells": celdas,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text(json.dumps(armar(), ensure_ascii=False, indent=1), encoding="utf-8")
    nb = json.loads(DESTINO.read_text(encoding="utf-8"))
    print(f"Generado: {DESTINO}")
    print(f"Celdas: {len(nb['cells'])} "
          f"({sum(c['cell_type'] == 'code' for c in nb['cells'])} de código)")
