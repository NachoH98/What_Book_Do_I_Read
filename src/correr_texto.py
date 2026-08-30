"""Compara texto vs. sin texto, y TF-IDF vs. palabras clave.

Uso:  python -m src.correr_texto
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, experimentos, modelo, texto, variables

# min_df y max_df son hiperparámetros, no constantes. Se prueban tres combinaciones:
# min_df filtra las palabras raras (ruido, nombres propios de un solo libro) y max_df
# las omnipresentes (que no distinguen nada).
REJILLA_TFIDF = [
    {"min_df": 5, "max_df": 0.5, "max_features": 2000},
    {"min_df": 20, "max_df": 0.3, "max_features": 2000},
    {"min_df": 50, "max_df": 0.2, "max_features": 1000},
]

PALABRAS_FINALES = 60


def main() -> pd.DataFrame:
    pd.set_option("display.width", 230)
    df = pd.read_pickle(config.CHECKPOINT_VARIABLES)
    y = df[config.TARGET].to_numpy()
    es_train = (~df[variables.COL_SPLIT]).to_numpy()
    print(f"Dataset: {df.shape[0]:,} × {df.shape[1]}\n")

    textos = texto.texto_por_fila(df)
    print("Nube de palabras sobre el texto limpio:")
    texto.nube_de_palabras(textos.drop_duplicates())

    print("\nPalabras más frecuentes tras el preprocesamiento (control visual):")
    frecuentes = pd.Series(" ".join(textos.drop_duplicates()).split()).value_counts()
    print("  " + ", ".join(frecuentes.head(25).index))

    # --- enfoque 1: TF-IDF, probando la rejilla ---
    mejor = None
    for parametros in REJILLA_TFIDF:
        matriz, vectorizador = texto.vectorizar(df, **parametros)
        etiqueta = (f"TF-IDF min_df={parametros['min_df']} "
                    f"max_df={parametros['max_df']} ({matriz.shape[1]} palabras)")
        print(f"\n· {etiqueta}")
        importancias = texto.importancias_del_texto(matriz, y, es_train)
        denso, palabras, imps = texto.seleccionar_columnas(
            matriz, vectorizador, importancias, PALABRAS_FINALES)
        con_texto = pd.concat([df.reset_index(drop=True), denso], axis=1)
        resultado = modelo.evaluar(con_texto, f"~ {etiqueta}")
        experimentos.registrar(resultado)
        print(f"  f1 {resultado['metrica_test']}  brecha {resultado['brecha']}")
        if mejor is None or resultado["metrica_test"] > mejor[0]["metrica_test"]:
            mejor = (resultado, con_texto, palabras, imps, etiqueta)

    # --- enfoque 2: columnas binarias por palabra clave ---
    print("\n· palabras clave binarias")
    binarias = texto.columnas_binarias(df)
    con_binarias = pd.concat([df.reset_index(drop=True), binarias.reset_index(drop=True)], axis=1)
    resultado_bin = modelo.evaluar(con_binarias, f"~ palabras clave binarias ({binarias.shape[1]})")
    experimentos.registrar(resultado_bin)
    print(f"  f1 {resultado_bin['metrica_test']}  brecha {resultado_bin['brecha']}")

    # --- el ganador ---
    resultado, con_texto, palabras, imps, etiqueta = mejor
    print(f"\nMejor configuración de texto: {etiqueta}")
    texto.grafico_palabras_seleccionadas(palabras, imps)
    print("  palabras seleccionadas:", ", ".join(palabras[:20]), "…")

    experimentos.registrar(modelo.evaluar(con_texto, "= CON TEXTO (mejor configuración)"))

    config.DIR_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    con_texto.to_pickle(config.CHECKPOINT_TEXTO)
    print(f"\nGuardado: {config.CHECKPOINT_TEXTO} — "
          f"{con_texto.shape[0]:,} × {con_texto.shape[1]} "
          f"({PALABRAS_FINALES} columnas de texto)")

    tabla = experimentos.tabla()
    print("\n" + "=" * 120)
    print(tabla.tail(8).to_string(index=False))
    return tabla


if __name__ == "__main__":
    main()
