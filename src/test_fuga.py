"""Verifica que ninguna variable nueva se haya calculado con información de test.

No es un test estructural ("¿la función recibe train?") sino causal: se da vuelta el
target de TODAS las filas de test, se recalculan las variables y se comprueba que
ninguna columna cambió. Si una columna se mueve cuando cambia el target del test, esa
columna estuvo mirando el test. Es la única forma de estar seguro, porque un bug de
indexado puede pasar cualquier revisión de código y no este test.

Uso:  python -m src.test_fuga
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, variables


def test_ninguna_variable_mira_el_test(df: pd.DataFrame, pipeline=None) -> list[str]:
    """Da vuelta el target del test y devuelve las columnas que cambiaron."""
    original = variables.aplicar(df, pipeline)

    saboteado = df.copy()
    es_test = saboteado[variables.COL_SPLIT]
    saboteado.loc[es_test, config.TARGET] = 1 - saboteado.loc[es_test, config.TARGET]
    alterado = variables.aplicar(saboteado, pipeline)

    nuevas = [c for c in original.columns if c not in df.columns]
    culpables = []
    for col in nuevas:
        a, b = original[col], alterado[col]
        if a.dtype.kind in "fc":
            distintas = not np.allclose(a.fillna(-999), b.fillna(-999), equal_nan=True)
        else:
            distintas = not a.equals(b)
        if distintas:
            culpables.append(col)
    return culpables


def test_los_perfiles_usan_solo_train(df: pd.DataFrame, pipeline=None) -> list[str]:
    """Comprueba que borrar las filas de test no cambia el perfil de las de train.

    Complementa al anterior: aquél detecta que el VALOR del target de test se cuele;
    éste detecta que la mera EXISTENCIA de filas de test altere un promedio o un
    conteo (por ejemplo, un value_counts() calculado sobre el dataset completo).
    """
    completo = variables.aplicar(df, pipeline)
    solo_train = variables.aplicar(df.loc[~df[variables.COL_SPLIT]].copy(), pipeline)

    nuevas = [c for c in completo.columns if c not in df.columns]
    filas_train = ~df[variables.COL_SPLIT]
    culpables = []
    for col in nuevas:
        if col not in solo_train.columns:
            culpables.append(f"{col} (no existe al correr sólo con train)")
            continue
        a = completo.loc[filas_train, col].reset_index(drop=True)
        b = solo_train[col].reset_index(drop=True)
        if a.dtype.kind in "fc":
            distintas = not np.allclose(a.fillna(-999), b.fillna(-999), equal_nan=True)
        else:
            distintas = not a.equals(b)
        if distintas:
            culpables.append(col)
    return culpables


def main() -> bool:
    base = pd.read_pickle(config.CHECKPOINT_LIMPIO)
    df = variables.dividir(base)
    print(f"Dataset: {len(df):,} filas · train {int((~df[variables.COL_SPLIT]).sum()):,} "
          f"· test {int(df[variables.COL_SPLIT].sum()):,}\n")

    pruebas = [
        ("Dar vuelta el target del test no mueve ninguna variable",
         test_ninguna_variable_mira_el_test),
        ("Borrar las filas de test no cambia los perfiles de train",
         test_los_perfiles_usan_solo_train),
    ]
    todo_bien = True
    # Las dos variantes se verifican: el leave-one-out toca la aritmética del perfil,
    # así que merece su propia prueba y no hereda la confianza de la otra.
    for nombre_pipeline, pipeline in [("cálculo directo", variables.PIPELINE),
                                      ("leave-one-out", variables.PIPELINE_LOO)]:
        print(f"— {nombre_pipeline}")
        for descripcion, prueba in pruebas:
            culpables = prueba(df, pipeline)
            estado = "PASA" if not culpables else "FALLA"
            print(f"  [{estado}] {descripcion}")
            if culpables:
                todo_bien = False
                for c in culpables:
                    print(f"           ← {c} se calculó con información de test")

    print("\n" + ("Ninguna variable nueva vio el test." if todo_bien
                  else "HAY FUGA: revisar las columnas de arriba."))
    return todo_bien


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
