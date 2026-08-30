"""Prueba los bloques de variables de a uno, contra el modelo congelado.

Uso:  python -m src.correr_variables
"""

from __future__ import annotations

import pandas as pd

from src import config, experimentos, modelo, variables

# Los bloques se miden con el mismo criterio que el pipeline elegido (leave-one-out).
# Medirlos con el cálculo directo daría números que no corresponden a lo que se entrega.
BLOQUES = [
    ("top-N de autor y editorial", [variables.top_n, variables.dummies_nuevas]),
    ("perfil del lector", [variables.top_n,
                           lambda d: variables.perfil_lector(d, loo=True),
                           lambda d: variables.perfil_lector_autor(d, loo=True)]),
    ("perfil del libro", [lambda d: variables.perfil_libro(d, loo=True)]),
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

    print("\n· todas las variables, cálculo directo (la variante descartada)")
    directo = variables.aplicar(df, variables.PIPELINE_DIRECTO)
    experimentos.registrar(modelo.evaluar(directo, "~ variables con cálculo directo (descartado)"))

    print("· todas las variables, leave-one-out (la elegida)")
    completo = variables.aplicar(df, variables.PIPELINE_LOO)
    experimentos.registrar(modelo.evaluar(completo, "= TODAS las variables (leave-one-out)"))

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
