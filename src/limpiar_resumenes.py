"""Lematiza los resúmenes únicos una sola vez y los cachea.

spaCy tarda ~10 minutos sobre los 44.112 resúmenes distintos. Rehacerlo en cada
experimento sería tirar el tiempo: se calcula una vez y se guarda.

Uso:  python -m src.limpiar_resumenes
"""

from __future__ import annotations

import time

import pandas as pd

from src import config, texto

DESTINO = config.DIR_CHECKPOINTS / "06_resumenes_limpios.pkl"


def main() -> pd.DataFrame:
    df = pd.read_pickle(config.CHECKPOINT_VARIABLES)
    unicos = (df.drop_duplicates("id_libro")[["id_libro", "resumen", "titulo"]]
              .reset_index(drop=True))
    print(f"{len(unicos):,} libros únicos ({unicos.resumen.notna().sum():,} con resumen)")

    # Título y resumen se concatenan: el título es corto pero muy denso en señal,
    # y no justifica una vectorización aparte.
    crudo = (unicos["titulo"].fillna("") + ". " + unicos["resumen"].fillna("")).tolist()

    inicio = time.perf_counter()
    unicos["texto_limpio"] = texto.limpiar(crudo)
    print(f"lematizado en {(time.perf_counter() - inicio) / 60:.1f} min")

    config.DIR_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    unicos[["id_libro", "texto_limpio"]].to_pickle(DESTINO)
    print(f"Guardado: {DESTINO}")
    vacios = (unicos["texto_limpio"].str.len() == 0).sum()
    print(f"textos vacíos tras limpiar: {vacios:,}")
    return unicos


if __name__ == "__main__":
    main()
