"""Corre el experimento #0: la base sin limpiar, contra el modelo congelado.

Es el número de referencia contra el que se comparan todos los cambios de las
ramas siguientes.

Uso:  python -m src.correr_base
"""

from __future__ import annotations

import pandas as pd

from src import config, experimentos, modelo

NOMBRE_BASE = "#0 — Base sin limpiar"


def main() -> pd.DataFrame:
    base = pd.read_pickle(config.CHECKPOINT_BASE)
    print(f"Dataset: {config.CHECKPOINT_BASE.name} — {len(base):,} filas × {base.shape[1]} columnas")

    X, _ = modelo.separar_x_y(base)
    descartadas = [c for c in base.columns if c not in X.columns]
    print(f"\nPredictoras que sobreviven ({X.shape[1]}): {list(X.columns)}")
    print(f"Descartadas ({len(descartadas)}): {descartadas}")
    print("  (las categóricas todavía no son dummies y `rating`/`gusto` son el target)")

    print(f"\nEntrenando el modelo congelado — métrica: {config.METRICA}")
    resultado = modelo.evaluar(base, NOMBRE_BASE)
    print(resultado)

    tabla = experimentos.registrar(resultado)
    print(f"\nTabla de experimentos → {config.TABLA_EXPERIMENTOS}")
    print(tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    main()
