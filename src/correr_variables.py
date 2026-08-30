"""Prueba los bloques de variables de a uno, contra el modelo congelado.

Uso:  python -m src.correr_variables
"""

from __future__ import annotations

import pandas as pd

from src import config, experimentos, modelo, variables

BLOQUES = [
    ("top-N de autor y editorial", [variables.top_n, variables.dummies_nuevas]),
    ("perfil del lector", [variables.top_n, variables.perfil_lector,
                           variables.perfil_lector_autor]),
    ("perfil del libro", [variables.perfil_libro]),
    ("antigüedad del libro y edad del lector", [variables.antiguedad_y_edad]),
    ("interacción región × editorial", [variables.top_n,
                                        variables.interaccion_region_editorial,
                                        variables.dummies_nuevas]),
]


def main() -> pd.DataFrame:
    pd.set_option("display.width", 230)
    base = pd.read_pickle(config.CHECKPOINT_LIMPIO)
    df = variables.dividir(base)
    print(f"Base limpia dividida: train {int((~df[variables.COL_SPLIT]).sum()):,} · "
          f"test {int(df[variables.COL_SPLIT].sum()):,}\n")

    # Referencia: el mismo dataset limpio, ya con el split explícito. Todo lo que
    # venga después se compara contra esto y no contra el #0, que no tenía variables.
    print("· referencia: dataset limpio con el split explícito")
    experimentos.registrar(modelo.evaluar(df, "= referencia: limpio con split explícito"))

    for etiqueta, funciones in BLOQUES:
        aislado = df
        for funcion in funciones:
            aislado = funcion(aislado)
        print(f"· bloque aislado: {etiqueta}  ({aislado.shape[1]} columnas)")
        experimentos.registrar(modelo.evaluar(aislado, f"~ sólo {etiqueta}"))

    print("\n· todas las variables juntas")
    completo = variables.aplicar(df)
    experimentos.registrar(modelo.evaluar(completo, "= TODAS las variables"))

    config.DIR_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    completo.to_pickle(config.CHECKPOINT_VARIABLES)
    print(f"\nGuardado: {config.CHECKPOINT_VARIABLES} — "
          f"{completo.shape[0]:,} filas × {completo.shape[1]} columnas")

    tabla = experimentos.tabla()
    print("\n" + "=" * 120)
    print(tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    main()
