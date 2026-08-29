"""Registro de experimentos: la tabla que la cátedra pide en el informe.

Cada fila es un cambio probado solo, contra el modelo congelado (CLAUDE.md 3.4).
La comparación siempre es contra el experimento #0, la base sin limpiar: eso es lo
que mide `delta`.
"""

from __future__ import annotations

import pandas as pd

from src import config

COLUMNAS = ["#", "cambio", "metrica_train", "metrica_test", "brecha",
            "delta", "delta_marginal", "filas", "columnas", "segundos", "queda"]

BASE = "#0"  # etiqueta de la fila base en la columna `queda`
# Las filas de resumen empiezan con "=". No son un paso más: comparar su delta
# marginal contra la fila de arriba no significa nada, porque la de arriba puede ser
# una variante descartada y no el paso anterior del pipeline.
PREFIJO_RESUMEN = "="


def _veredicto(delta_marginal) -> str:
    """sí / no / ruido, según el delta marginal contra la banda medida del instrumento."""
    if pd.isna(delta_marginal):
        return ""
    if abs(delta_marginal) < config.UMBRAL_RUIDO:
        return "ruido"
    return "sí" if delta_marginal > 0 else "no"


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

    Hay dos deltas y miden cosas distintas:

    - `delta` compara contra el #0 y responde "¿el pipeline hasta acá es mejor que la
      base?". Es acumulado.
    - `delta_marginal` compara contra el paso anterior y responde "¿este cambio, solo,
      aportó algo?". Es el que corresponde a ceteris paribus (CLAUDE.md 3.4), y el
      que decide `queda`.

    `queda` marca "ruido" cuando el delta marginal cae dentro de la banda medida del
    instrumento (config.UMBRAL_RUIDO): ahí no se puede afirmar ni que sirve ni que no.
    La decisión final es del informe, no de esta función.
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
        "delta_marginal": None,
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
        nueva["delta_marginal"] = nueva["metrica_test"].diff().round(4)
        nueva["queda"] = nueva["delta_marginal"].apply(_veredicto)
        nueva.loc[nueva["#"] == 0, ["delta_marginal", "queda"]] = [None, BASE]
        resumen = nueva["cambio"].str.startswith(PREFIJO_RESUMEN)
        nueva.loc[resumen, ["delta_marginal", "queda"]] = [None, "final"]

    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    nueva[COLUMNAS].to_csv(config.TABLA_EXPERIMENTOS, index=False)
    return nueva[COLUMNAS].reset_index(drop=True)
