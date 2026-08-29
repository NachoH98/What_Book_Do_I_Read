"""Lectura y unión de las tres tablas. Acá no se limpia nada: sólo se lee, se une
y se construye el target.
"""

from __future__ import annotations

import pandas as pd

from src import config


# --------------------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------------------

def cargar_tablas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee los tres CSV tal cual vienen y los devuelve por separado, sin transformar.

    No se castean tipos ni se parsean fechas a propósito: cualquier conversión es una
    decisión de limpieza y va en la rama que corresponde.
    """
    libros = pd.read_csv(config.CSV_LIBROS, low_memory=False)
    lectores = pd.read_csv(config.CSV_LECTORES, low_memory=False)
    opiniones = pd.read_csv(config.CSV_OPINIONES, low_memory=False)
    return libros, lectores, opiniones


# --------------------------------------------------------------------------------------
# Perfilado
# --------------------------------------------------------------------------------------

def porcentaje_nulos(df: pd.DataFrame) -> pd.Series:
    """% de nulos por columna, ordenado de mayor a menor."""
    return (df.isna().mean() * 100).round(2).sort_values(ascending=False)


def perfilar_tablas(
    libros: pd.DataFrame,
    lectores: pd.DataFrame,
    opiniones: pd.DataFrame,
) -> None:
    """Imprime filas, tipos, % de nulos por columna e IDs únicos de cada tabla."""
    # Para cada tabla: sus columnas de ID, y la clave que debería identificar una fila.
    tablas = [
        ("libros", libros, [config.COL_ID_LIBRO], [config.COL_ID_LIBRO]),
        ("lectores", lectores, [config.COL_ID_LECTOR], [config.COL_ID_LECTOR]),
        ("opiniones", opiniones, [config.COL_ID_LECTOR, config.COL_ID_LIBRO],
         [config.COL_ID_LECTOR, config.COL_ID_LIBRO]),
    ]

    for nombre, df, cols_id, clave in tablas:
        print(f"\n### Tabla `{nombre}` — {len(df):,} filas × {df.shape[1]} columnas")

        perfil = pd.DataFrame(
            {
                "tipo": df.dtypes.astype(str),
                "% nulos": (df.isna().mean() * 100).round(2),
                "valores únicos": df.nunique(),
            }
        )
        print(perfil.to_string())

        for col in cols_id:
            print(f"  IDs únicos en `{col}`: {df[col].nunique():,}")
        duplicados = int(df.duplicated(subset=clave).sum())
        print(f"  Filas duplicadas por la clave {clave}: {duplicados:,}"
              f"{'  <-- el merge podría multiplicar filas' if duplicados else ''}")


# --------------------------------------------------------------------------------------
# Unión
# --------------------------------------------------------------------------------------

def unir(
    libros: pd.DataFrame,
    lectores: pd.DataFrame,
    opiniones: pd.DataFrame,
    how: str = config.HOW_UNION,
) -> pd.DataFrame:
    """Une las tres tablas dejando a `opiniones` en el centro: una fila = una opinión.

    `how` decide qué pasa con una opinión cuyo libro o lector no existe en las otras
    tablas: con "left" la fila se conserva con todos los atributos en nulo, con "inner"
    se descarta. Ninguna de las dos tablas laterales tiene IDs duplicados, así que el
    merge no puede multiplicar filas.

    Las columnas `genero` de libros y de lectores colisionan: quedan como
    `genero_libro` y `genero_lector`.
    """
    return (
        opiniones
        .merge(libros, on=config.COL_ID_LIBRO, how=how)
        .merge(lectores, on=config.COL_ID_LECTOR, how=how, suffixes=("_libro", "_lector"))
    )


# --------------------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------------------

def construir_target(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta los rating 6 y recién después mapea >= 7 a 1 y <= 5 a 0.

    El orden importa: el 6 es un rating gris que no aporta información, así que sale
    del dataset antes de binarizar. Se conserva la columna `rating` para trazabilidad,
    pero queda prohibida como predictora (CLAUDE.md 3.3): contiene el target.
    """
    rating = df[config.COL_RATING]
    mask = rating.notna() & rating.ne(config.RATING_DESCARTADO)

    return (
        df.loc[mask]
        .assign(**{config.TARGET: (rating[mask] >= config.RATING_MIN_GUSTO).astype("int8")})
        .reset_index(drop=True)
    )
