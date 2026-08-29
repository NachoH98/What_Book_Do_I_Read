"""Registro de experimentos: la tabla que la cátedra pide en el informe.

Cada fila es un cambio probado solo, contra el modelo congelado (CLAUDE.md 3.4).
La comparación siempre es contra el experimento #0, la base sin limpiar: eso es lo
que mide `delta`.
"""

from __future__ import annotations

import pandas as pd

from src import config

COLUMNAS = ["#", "cambio", "metrica_train", "metrica_test", "brecha",
            "delta", "filas", "columnas", "segundos", "queda"]

BASE = "#0"  # etiqueta de la fila base en la columna `queda`


def tabla() -> pd.DataFrame:
    """Devuelve la tabla de experimentos ordenada por número. Vacía si no existe."""
    if not config.TABLA_EXPERIMENTOS.is_file():
        return pd.DataFrame(columns=COLUMNAS)
    return pd.read_csv(config.TABLA_EXPERIMENTOS).sort_values("#").reset_index(drop=True)


def registrar(resultado: dict) -> pd.DataFrame:
    """Agrega (o actualiza) una fila en resultados/tabla_experimentos.csv.

    Si el experimento ya estaba registrado con el mismo nombre, se pisa la fila en
    lugar de duplicarla: volver a correr un experimento es normal, tener dos filas
    con el mismo nombre y números distintos no.

    `delta` es la diferencia contra el experimento #0 y `queda` es la sugerencia
    que se desprende de ella: si el cambio no mejora la métrica, no queda. La
    decisión final es del informe, no de esta función.
    """
    previa = tabla()
    ya_estaba = previa["cambio"] == resultado["cambio"]

    numero = int(previa.loc[ya_estaba, "#"].iloc[0]) if ya_estaba.any() else len(previa)

    fila = {
        "#": numero,
        "cambio": resultado["cambio"],
        "metrica_train": resultado["metrica_train"],
        "metrica_test": resultado["metrica_test"],
        "brecha": resultado["brecha"],
        "delta": None,
        "filas": resultado["filas"],
        "columnas": resultado["columnas"],
        "segundos": resultado["segundos"],
        "queda": None,
    }

    nueva = previa[~ya_estaba] if ya_estaba.any() else previa
    nueva = pd.concat([nueva, pd.DataFrame([fila])], ignore_index=True).sort_values("#")

    # El delta se recalcula sobre toda la tabla: si se vuelve a correr la base,
    # todos los deltas que dependen de ella tienen que moverse con ella.
    base = nueva.loc[nueva["#"] == 0, "metrica_test"]
    if not base.empty:
        referencia = float(base.iloc[0])
        nueva["delta"] = (nueva["metrica_test"] - referencia).round(4)
        nueva["queda"] = nueva["delta"].apply(lambda d: "sí" if d > 0 else "no")
        nueva.loc[nueva["#"] == 0, "queda"] = BASE

    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    nueva[COLUMNAS].to_csv(config.TABLA_EXPERIMENTOS, index=False)
    return nueva[COLUMNAS].reset_index(drop=True)
