"""Prueba las limpiezas de a una, en forma acumulativa, contra el modelo congelado.

Cada paso agrega UNA transformación sobre el anterior y produce una fila de la tabla
de experimentos con su delta contra el #0. Ceteris paribus (CLAUDE.md 3.4).

Detalle de método: antes de evaluar, cada etapa pasa por `crear_dummies`. El modelo
congelado sólo mira columnas numéricas, así que sin ese paso las transformaciones
sobre categóricas —región, género del lector, género literario— darían delta cero, no
porque no sirvan sino porque el instrumento no las ve. Las dummies acá son parte del
aparato de medición; en el pipeline final son el último paso, como manda CLAUDE.md 3.5.

Uso:  python -m src.correr_limpieza
"""

from __future__ import annotations

import pandas as pd

from src import config, experimentos, limpieza, modelo

# Cada entrada es (etiqueta, función). Se aplican de forma acumulativa.
# La fila de referencia separa el efecto de las dummies del de la primera limpieza.
# Sin ella, el paso 1 se llevaría el crédito de un salto que en realidad es de las
# dummies del género literario.
REFERENCIA = ("(referencia) sólo dummies, sin limpiar", lambda df: df)

PASOS = [
    ("filtrar opiniones sin rating", limpieza.filtrar_sin_target),
    ("descartar libros inexistentes", limpieza.descartar_no_libros),
    ("año de nacimiento imposible → nulo", limpieza.anio_nacimiento_a_nulo),
    ("año de edición ilegible → nulo", limpieza.anio_edicion_a_nulo),
    ("normalizar vive_en → región", limpieza.normalizar_vive_en),
    ("imputar género del lector", limpieza.imputar_genero_lector),
    ("imputar categóricas con Desconocido", limpieza.imputar_categoricas),
]

# Se prueba aparte, sobre el pipeline completo: anular el 7,4% de una predictora es
# una decisión distinta a corregir basura.
APARTE = [("año de edición posterior a la opinión → nulo", limpieza.edicion_posterior_a_nulo)]


def evaluar_etapa(df: pd.DataFrame, etiqueta: str) -> dict:
    """Aplica dummies y mide. Las dummies son del aparato de medición, no del paso."""
    return modelo.evaluar(limpieza.crear_dummies(df), etiqueta)


def main() -> pd.DataFrame:
    pd.set_option("display.width", 220)
    base = pd.read_pickle(config.CHECKPOINT_BASE)
    print(f"Base: {len(base):,} filas × {base.shape[1]} columnas\n")

    # La tabla se reconstruye entera: los deltas marginales dependen del orden y de
    # los vecinos, así que una tabla con filas de corridas distintas no sería coherente.
    config.TABLA_EXPERIMENTOS.unlink(missing_ok=True)

    print("#0 base sin limpiar (sin dummies: como la ve el modelo hoy)")
    experimentos.registrar(modelo.evaluar(base, "Base sin limpiar"))

    etiqueta, funcion = REFERENCIA
    print(f"#1 {etiqueta}")
    experimentos.registrar(evaluar_etapa(funcion(base), etiqueta))

    acumulado = base
    for numero, (etiqueta, funcion) in enumerate(PASOS, start=2):
        antes = len(acumulado)
        acumulado = funcion(acumulado)
        print(f"#{numero} +{etiqueta}  ({antes:,} → {len(acumulado):,} filas)")
        experimentos.registrar(evaluar_etapa(acumulado, f"+{etiqueta}"))

    numero = len(PASOS) + 2
    for etiqueta, funcion in APARTE:
        print(f"#{numero} +{etiqueta}  (aparte, sobre el pipeline completo)")
        experimentos.registrar(evaluar_etapa(funcion(acumulado), f"+{etiqueta}"))
        numero += 1

    # El checkpoint lleva SÓLO las transformaciones que quedaron en el PIPELINE.
    print("\nPipeline final (sólo lo que quedó):")
    for funcion in limpieza.PIPELINE:
        print(f"  · {funcion.__name__}")
    limpio = limpieza.aplicar(base)
    experimentos.registrar(modelo.evaluar(limpio, "= PIPELINE FINAL (sólo lo que quedó)"))
    config.DIR_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    limpio.to_pickle(config.CHECKPOINT_LIMPIO)
    print(f"\nGuardado: {config.CHECKPOINT_LIMPIO} — "
          f"{limpio.shape[0]:,} filas × {limpio.shape[1]} columnas")

    tabla = experimentos.tabla()
    print("\n" + "=" * 110)
    print(f"TABLA DE EXPERIMENTOS — métrica: {config.METRICA} sobre la clase {config.CLASE_MEDIDA}")
    print("=" * 110)
    print(tabla.to_string(index=False))
    print(f"\nBanda de ruido del instrumento: ±{config.UMBRAL_RUIDO}. "
          f"`queda` se decide con el delta MARGINAL, no con el acumulado.")
    return tabla


if __name__ == "__main__":
    main()
