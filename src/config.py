"""Configuración global del TP: rutas, semilla, métrica de decisión y cortes de rating.

Todo lo que otro módulo necesite parametrizar vive acá. Nadie más hardcodea rutas
ni números mágicos.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------

def _raiz() -> Path:
    """Raíz del proyecto, tanto corriendo como módulo como dentro del notebook.

    En el notebook no existe `__file__` y el directorio de trabajo puede ser
    `notebooks/`, así que se sube hasta encontrar el CLAUDE.md. En Colab, donde no
    está, cae en el directorio de trabajo y todo cuelga de ahí.
    """
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    actual = Path.cwd().resolve()
    for candidata in (actual, *actual.parents):
        if (candidata / "CLAUDE.md").is_file():
            return candidata
    return actual


RAIZ = _raiz()

# Los CSV no están versionados. Por defecto se buscan en data/; se puede apuntar a
# otro lado con la variable de entorno QLL_DATA_DIR (útil en Colab, donde los datos
# quedan montados en Drive).
DIR_DATOS = Path(os.environ.get("QLL_DATA_DIR", RAIZ / "data"))
DIR_CHECKPOINTS = Path(os.environ.get("QLL_CHECKPOINTS_DIR", RAIZ / "checkpoints"))
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"

# Nombres de archivo tal como los entregó la cátedra. La tabla de opiniones viene
# con el nombre "interacciones.csv".
ARCHIVO_LIBROS = os.environ.get("QLL_CSV_LIBROS", "libros.csv")
ARCHIVO_LECTORES = os.environ.get("QLL_CSV_LECTORES", "lectores.csv")
ARCHIVO_OPINIONES = os.environ.get("QLL_CSV_OPINIONES", "interacciones.csv")


def resolver_csv(nombre: str) -> Path:
    """Devuelve la ruta del CSV: primero en DIR_DATOS, si no está, en la raíz del repo.

    El fallback a la raíz existe porque los CSV de la cátedra hoy están ahí; la ruta
    canónica sigue siendo data/. Si no aparece en ninguna de las dos, devuelve la
    canónica para que el error de lectura indique dónde había que ponerlo.
    """
    for candidata in (DIR_DATOS / nombre, RAIZ / nombre):
        if candidata.is_file():
            return candidata
    return DIR_DATOS / nombre


CSV_LIBROS = resolver_csv(ARCHIVO_LIBROS)
CSV_LECTORES = resolver_csv(ARCHIVO_LECTORES)
CSV_OPINIONES = resolver_csv(ARCHIVO_OPINIONES)

CHECKPOINT_BASE = DIR_CHECKPOINTS / "01_base.pkl"
CHECKPOINT_LIMPIO = DIR_CHECKPOINTS / "04_limpio.pkl"

# --------------------------------------------------------------------------------------
# Reproducibilidad y evaluación
# --------------------------------------------------------------------------------------

SEED = 42

# Métrica única de decisión (CLAUDE.md 3.2). El dataset está desbalanceado ~80/20,
# así que accuracy está prohibida. Todas las comparaciones entre experimentos se
# hacen con esta y sólo con esta.
METRICA = "f1"

# Sobre qué clase se calcula la métrica. Se mide la 0 ("no le gustó"), que es la
# minoritaria y la informativa. Con la clase 1, predecir siempre "le gustó" da
# f1 = 0.9123 sin aprender nada, y ningún modelo razonable lo supera: la métrica
# quedaría dominada por la proporción de clases en vez de medir señal. Sobre la
# clase 0 ese clasificador trivial da f1 = 0, así que toda mejora es real.
CLASE_MEDIDA = 0

# Proporción del test en el split. El split es estratificado porque las clases
# están ~84/16 y un split al azar podría desbalancearlas todavía más.
TEST_SIZE = 0.25

# Hiperparámetros del modelo congelado de evaluación (CLAUDE.md 3.4). No se tocan
# hasta la rama de optimización: si el modelo cambia entre experimentos, la
# comparación entre ellos no mide el cambio, mide el modelo.
MODELO_CONGELADO = {
    "n_estimators": 300,
    "max_depth": 8,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
}

# Valor con el que se rellenan los nulos que sobrevivan, sólo para que el modelo
# pueda correr sobre un dataset sucio. Es un centinela, no una imputación: imputar
# es una decisión de la rama de limpieza y el instrumento de medición no la toma
# por su cuenta.
RELLENO_NULOS = -1

TABLA_EXPERIMENTOS = DIR_RESULTADOS / "tabla_experimentos.csv"

# Banda de ruido del instrumento, medida: el mismo dataset evaluado con 6 semillas de
# split distintas da σ = 0.0024, así que 2σ ≈ 0.005. Un delta más chico que esto no se
# distingue del azar del split y no alcanza para declarar que un cambio sirve.
UMBRAL_RUIDO = 0.005

# --------------------------------------------------------------------------------------
# Definición del target
# --------------------------------------------------------------------------------------

COL_RATING = "rating"
TARGET = "gusto"

RATING_MIN_GUSTO = 7      # rating >= 7  -> gusto = 1
RATING_MAX_NO_GUSTO = 5   # rating <= 5  -> gusto = 0
RATING_DESCARTADO = 6     # rating == 6  -> la fila se elimina (rating gris)

# --------------------------------------------------------------------------------------
# Unión de tablas
# --------------------------------------------------------------------------------------

COL_ID_LIBRO = "id_libro"
COL_ID_LECTOR = "id_lector"

# Cómo se resuelven las opiniones que apuntan a un libro o a un lector inexistente.
# Se decide con el diagnóstico de src/diagnostico.py.
HOW_UNION = "left"
