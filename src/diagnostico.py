"""Diagnóstico de la unión de las tres tablas. Material para el informe.

Corre la carga, el perfilado, la unión y la construcción del target, e imprime:
cobertura de libros y lectores, opiniones huérfanas, % de nulos antes y después
del merge, comparación left vs inner y distribución de clases.

Uso:  python -m src.diagnostico [--how left|inner]
      python -m src.diagnostico | tee resultados/01_diagnostico_union.txt
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import carga, config

COLS_LIBROS = ["titulo", "autor", "genero_libro", "editorial", "anio_edicion",
               "isbn", "resumen", "img_src"]
COLS_LECTORES = ["nombre", "genero_lector", "vive_en", "nacimiento"]
COLS_OPINIONES = ["id_lector", "id_libro", "fecha", "rating"]

# Nombre en la tabla de origen -> nombre después del merge (la colisión de `genero`).
RENOMBRES = {"genero_libro": "genero", "genero_lector": "genero"}


def titulo(texto: str) -> None:
    print("\n" + "=" * 88)
    print(texto)
    print("=" * 88)


# --------------------------------------------------------------------------------------
# 1. Cobertura: cuánto de cada tabla lateral usa realmente el dataset
# --------------------------------------------------------------------------------------

def cobertura(libros: pd.DataFrame, lectores: pd.DataFrame, opiniones: pd.DataFrame) -> pd.DataFrame:
    """Cuántas entidades hay en cada tabla y cuántas aparecen en al menos una opinión."""
    filas = []
    for nombre, df, col in [("libros", libros, config.COL_ID_LIBRO),
                            ("lectores", lectores, config.COL_ID_LECTOR)]:
        en_tabla = df[col].nunique()
        referenciados = set(opiniones[col].unique())
        con_opinion = df[col].isin(referenciados).sum()
        filas.append({
            "tabla": nombre,
            "en la tabla": en_tabla,
            "con al menos una opinión": int(con_opinion),
            "% usado": round(100 * con_opinion / en_tabla, 2),
            "sin ninguna opinión": int(en_tabla - con_opinion),
            "% descartado por el merge": round(100 * (en_tabla - con_opinion) / en_tabla, 2),
        })
    return pd.DataFrame(filas).set_index("tabla")


def huerfanas(libros: pd.DataFrame, lectores: pd.DataFrame, opiniones: pd.DataFrame) -> pd.DataFrame:
    """Opiniones que apuntan a un libro o a un lector que no existe en su tabla."""
    sin_libro = ~opiniones[config.COL_ID_LIBRO].isin(set(libros[config.COL_ID_LIBRO]))
    sin_lector = ~opiniones[config.COL_ID_LECTOR].isin(set(lectores[config.COL_ID_LECTOR]))
    total = len(opiniones)

    filas = [
        ("libro inexistente", int(sin_libro.sum())),
        ("lector inexistente", int(sin_lector.sum())),
        ("libro Y lector inexistentes", int((sin_libro & sin_lector).sum())),
        ("al menos una de las dos (se pierden con inner)", int((sin_libro | sin_lector).sum())),
    ]
    return pd.DataFrame(
        [{"caso": c, "opiniones": n, "% del total": round(100 * n / total, 3)} for c, n in filas]
    ).set_index("caso")


# --------------------------------------------------------------------------------------
# 2. Nulos antes y después del merge
# --------------------------------------------------------------------------------------

def tabla_nulos(libros, lectores, opiniones, unidos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """% de nulos por columna en la tabla de origen y en el dataset unido (left e inner).

    La columna intermedia — la tabla de origen restringida a las entidades que sí
    aparecen en opiniones — es la que separa las dos causas de nulos: los registros
    vacíos que nunca vamos a ver, y los que el merge sí arrastra al dataset.
    """
    con_opinion = {
        "libros": libros[libros[config.COL_ID_LIBRO].isin(set(opiniones[config.COL_ID_LIBRO]))],
        "lectores": lectores[lectores[config.COL_ID_LECTOR].isin(set(opiniones[config.COL_ID_LECTOR]))],
        "opiniones": opiniones,
    }
    origen = {"libros": libros, "lectores": lectores, "opiniones": opiniones}

    filas = []
    for tabla, columnas in [("opiniones", COLS_OPINIONES),
                            ("libros", COLS_LIBROS),
                            ("lectores", COLS_LECTORES)]:
        for col in columnas:
            col_origen = RENOMBRES.get(col, col)
            antes = 100 * origen[tabla][col_origen].isna().mean()
            filtrado = 100 * con_opinion[tabla][col_origen].isna().mean()
            fila = {
                "tabla": tabla,
                "columna": col,
                "% nulos ANTES (tabla completa)": round(antes, 2),
                "% nulos (sólo con opiniones)": round(filtrado, 2),
            }
            for nombre_how, df in unidos.items():
                fila[f"% nulos DESPUÉS ({nombre_how})"] = round(100 * df[col].isna().mean(), 2)
            fila["reducción (pp)"] = round(antes - 100 * unidos["left"][col].isna().mean(), 2)
            filas.append(fila)

    return pd.DataFrame(filas).set_index(["tabla", "columna"])


def resumen_nulos_libros(libros: pd.DataFrame, unidos: dict[str, pd.DataFrame]) -> None:
    """El número del informe: cuánto del 60% de nulos de `libros` nos toca limpiar."""
    cols_origen = [RENOMBRES.get(c, c) for c in COLS_LIBROS]

    celdas_tabla = libros[cols_origen].size
    nulas_tabla = int(libros[cols_origen].isna().sum().sum())

    # Un libro está "vacío" si no tiene ningún atributo cargado.
    vacios = libros[cols_origen].isna().all(axis=1)
    print(f"Celdas nulas en las columnas de `libros` (tabla completa): "
          f"{nulas_tabla:,} de {celdas_tabla:,}  ({100 * nulas_tabla / celdas_tabla:.2f} %)")
    print(f"Libros sin ningún atributo cargado (fila entera vacía): "
          f"{int(vacios.sum()):,} de {len(libros):,}  ({100 * vacios.mean():.2f} %)")

    for nombre_how, df in unidos.items():
        nulas = int(df[COLS_LIBROS].isna().sum().sum())
        print(f"Nulos en columnas de libros dentro del dataset unido ({nombre_how}): "
              f"{nulas:,} de {df[COLS_LIBROS].size:,}  ({100 * nulas / df[COLS_LIBROS].size:.2f} %)")


# --------------------------------------------------------------------------------------
# 3. left vs inner
# --------------------------------------------------------------------------------------

def comparar_how(unidos: dict[str, pd.DataFrame], opiniones: pd.DataFrame) -> pd.DataFrame:
    """Costo y beneficio de cada `how`, en filas y en calidad de las columnas."""
    filas = []
    for nombre_how, df in unidos.items():
        con_target = carga.construir_target(df)
        nulos_libros = 100 * df[COLS_LIBROS].isna().mean().mean()
        nulos_lectores = 100 * df[COLS_LECTORES].isna().mean().mean()
        filas.append({
            "how": nombre_how,
            "filas tras el merge": len(df),
            "filas tras quitar rating 6": len(con_target),
            "% de opiniones conservadas": round(100 * len(df) / len(opiniones), 3),
            "% nulos promedio (cols. libros)": round(nulos_libros, 2),
            "% nulos promedio (cols. lectores)": round(nulos_lectores, 2),
            "% clase 1 (gustó)": round(100 * con_target[config.TARGET].mean(), 2),
        })
    return pd.DataFrame(filas).set_index("how")


def target_en_huerfanas(base_left: pd.DataFrame) -> pd.DataFrame:
    """¿Las opiniones huérfanas tienen un target distinto? Si no, dropearlas no sesga."""
    df = carga.construir_target(base_left)
    es_huerfana = df["titulo"].isna() | df["nombre"].isna()
    resumen = (
        df.assign(grupo=es_huerfana.map({True: "huérfanas", False: "con libro y lector"}))
        .groupby("grupo")[config.TARGET]
        .agg(filas="size", **{"% gustó": lambda s: round(100 * s.mean(), 2)})
    )
    return resumen


# --------------------------------------------------------------------------------------
# 4. Distribución de clases
# --------------------------------------------------------------------------------------

def distribucion_clases(opiniones: pd.DataFrame, base: pd.DataFrame) -> None:
    print("Ratings en la tabla de opiniones (antes de construir el target):")
    conteo = opiniones[config.COL_RATING].value_counts().sort_index()
    tabla = pd.DataFrame({
        "opiniones": conteo,
        "% del total": (100 * conteo / len(opiniones)).round(2),
        "clase": [
            "1 (gustó)" if r >= config.RATING_MIN_GUSTO
            else "DESCARTADO" if r == config.RATING_DESCARTADO
            else "0 (no gustó)"
            for r in conteo.index
        ],
    })
    print(tabla.to_string())

    descartadas = int((opiniones[config.COL_RATING] == config.RATING_DESCARTADO).sum())
    print(f"\nFilas eliminadas por rating == 6: {descartadas:,} "
          f"({100 * descartadas / len(opiniones):.2f} % de las opiniones)")

    print(f"\nDistribución final de `{config.TARGET}` ({len(base):,} filas):")
    dist = base[config.TARGET].value_counts().sort_index()
    print(pd.DataFrame({
        "filas": dist,
        "%": (100 * dist / len(base)).round(2),
    }).to_string())
    ratio = dist.max() / dist.min()
    print(f"\nRatio de desbalanceo: {ratio:.2f} a 1 — desbalanceo clásico, no extremo.")


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def main(how: str = config.HOW_UNION) -> pd.DataFrame:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    titulo("0. LECTURA DE LAS TRES TABLAS")
    print(f"libros:    {config.CSV_LIBROS}")
    print(f"lectores:  {config.CSV_LECTORES}")
    print(f"opiniones: {config.CSV_OPINIONES}")
    libros, lectores, opiniones = carga.cargar_tablas()

    titulo("1. PERFIL DE CADA TABLA POR SEPARADO")
    carga.perfilar_tablas(libros, lectores, opiniones)

    titulo("2. COBERTURA: CUÁNTO DE CADA TABLA USA EL DATASET")
    print(cobertura(libros, lectores, opiniones).to_string())
    print("\nOpiniones huérfanas (referencian un ID que no existe):")
    print(huerfanas(libros, lectores, opiniones).to_string())

    unidos = {
        "left": carga.unir(libros, lectores, opiniones, how="left"),
        "inner": carga.unir(libros, lectores, opiniones, how="inner"),
    }

    titulo("3. NULOS ANTES Y DESPUÉS DEL MERGE")
    nulos = tabla_nulos(libros, lectores, opiniones, unidos)
    print(nulos.to_string())

    titulo("4. EL 60% DE NULOS DE `libros` NO ES UN PROBLEMA A LIMPIAR")
    resumen_nulos_libros(libros, unidos)

    titulo("5. left vs inner")
    comparacion = comparar_how(unidos, opiniones)
    print(comparacion.to_string())
    print("\n¿Las opiniones huérfanas tienen otro comportamiento de target?")
    print(target_en_huerfanas(unidos["left"]).to_string())

    base = carga.construir_target(unidos[how])

    titulo(f"6. DISTRIBUCIÓN DE CLASES (dataset construido con how='{how}')")
    distribucion_clases(opiniones, base)

    titulo("7. SALIDAS")
    # Las dos tablas van al informe, así que se guardan en resultados/ (sí versionado).
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    nulos.to_csv(config.DIR_RESULTADOS / "01_nulos_antes_despues.csv")
    comparacion.to_csv(config.DIR_RESULTADOS / "01_comparacion_left_inner.csv")
    print(f"Guardado: {config.DIR_RESULTADOS / '01_nulos_antes_despues.csv'}")
    print(f"Guardado: {config.DIR_RESULTADOS / '01_comparacion_left_inner.csv'}")

    config.DIR_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    base.to_pickle(config.CHECKPOINT_BASE)
    print(f"Guardado: {config.CHECKPOINT_BASE}")
    print(f"Dimensiones: {base.shape[0]:,} filas × {base.shape[1]} columnas  "
          f"(how='{how}', {config.CHECKPOINT_BASE.stat().st_size / 1e6:.1f} MB)")
    print(f"Columnas: {list(base.columns)}")
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--how", choices=["left", "inner"], default=config.HOW_UNION,
                        help="cómo resolver las opiniones huérfanas al unir")
    main(parser.parse_args().how)
