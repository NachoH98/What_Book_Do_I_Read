"""Modelo congelado de evaluación: el instrumento con el que se mide cada cambio.

La regla del curso es ceteris paribus (CLAUDE.md 3.4): un cambio por vez contra el
mismo modelo. Por eso los hiperparámetros están fijos en `config.MODELO_CONGELADO` y
no se tocan hasta la rama de optimización. Si el modelo cambiara entre experimentos,
la diferencia de métrica no mediría el cambio, mediría el modelo.
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src import config

# Una sola métrica de decisión (CLAUDE.md 3.2). `accuracy` no está y no va a estar:
# con clases ~84/16, predecir siempre 1 da 84% y no significa nada.
METRICAS = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def es_derivada_del_target(columna: str) -> bool:
    """¿La columna contiene el target y por lo tanto no puede ser predictora?

    `rating` es el target antes de binarizar y `gusto` es el target. Cualquier
    columna derivada de ellos (`rating_medio_del_lector`, `gusto_previo`, ...)
    tambien queda afuera: entrenar con ellas es fuga de información (CLAUDE.md 3.3).
    """
    nombre = columna.lower()
    return config.COL_RATING in nombre or config.TARGET in nombre


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve X (predictoras numéricas, sin nulos) e y (el target).

    Se queda sólo con las columnas numéricas y rellena con un centinela para que el
    modelo pueda correr aunque el dataset todavía esté sucio. Ni el descarte de las
    categóricas ni el relleno son decisiones de modelado: son el mínimo necesario
    para que el instrumento funcione sobre datos crudos. Convertir categóricas en
    dummies e imputar con criterio son cambios que se miden con esta misma función.
    """
    if config.TARGET not in df.columns:
        raise ValueError(f"El DataFrame no tiene la columna `{config.TARGET}`.")

    y = df[config.TARGET]
    predictoras = [c for c in df.columns if not es_derivada_del_target(c)]
    X = df[predictoras].select_dtypes(include="number").fillna(config.RELLENO_NULOS)

    if X.empty or X.shape[1] == 0:
        raise ValueError("No quedó ninguna columna numérica para entrenar.")
    return X, y


def modelo_congelado() -> RandomForestClassifier:
    """El clasificador de evaluación, siempre con los mismos hiperparámetros."""
    return RandomForestClassifier(**config.MODELO_CONGELADO)


def evaluar(df: pd.DataFrame, nombre_experimento: str) -> dict:
    """Entrena el modelo congelado sobre `df` y devuelve las métricas del experimento.

    La brecha (test − train) es tan importante como la métrica: si el test mejora
    pero la brecha se agranda, el cambio está sobreajustando y no generalizando.
    """
    inicio = time.perf_counter()

    X, y = separar_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=y,  # las clases están ~84/16: sin estratificar el split las corre
    )

    modelo = modelo_congelado().fit(X_train, y_train)
    metrica = METRICAS[config.METRICA]
    m_train = metrica(y_train, modelo.predict(X_train))
    m_test = metrica(y_test, modelo.predict(X_test))

    # Las claves son genéricas (`metrica_*`) y no `f1_*`: cuál es la métrica se
    # declara una sola vez, en config.METRICA, y no se repite en cada columna.
    return {
        "cambio": nombre_experimento,
        "metrica": config.METRICA,
        "metrica_train": round(m_train, 4),
        "metrica_test": round(m_test, 4),
        "brecha": round(m_test - m_train, 4),
        "filas": len(X),
        "columnas": X.shape[1],
        "segundos": round(time.perf_counter() - inicio, 1),
    }
