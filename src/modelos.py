"""Desbalanceo y comparación de modelos. Acá sí se toca el modelo.

Sobre el tamaño de las corridas (CLAUDE.md 5): la búsqueda de hiperparámetros y la
validación cruzada se hacen sobre una muestra estratificada de train, no sobre las
292.131 filas completas. Un GridSearch de GradientBoosting sobre el total serían
horas. Los modelos finales sí se entrenan con TODO el train y se evalúan contra el
test completo, que es el número que va a la tabla.

Uso:  python -m src.modelos
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as PipelineImb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from src import config, modelo, variables

MUESTRA_BUSQUEDA = 60_000   # para RandomizedSearch
MUESTRA_CV = 25_000         # para la validación cruzada en 10 partes
FOLDS = 10

# Muestra para el ENTRENAMIENTO FINAL de cada modelo. La evaluación sigue siendo contra
# el TEST COMPLETO (97.377 filas), así que el número de la tabla es comparable con todo
# lo anterior; lo que se reduce es cuántas filas ve el modelo al entrenar.
#
# CLAUDE.md 5 pide que la versión final corra con el 100%, y esto no lo cumple. El
# motivo es concreto: KNN calcula distancias contra TODAS las filas de entrenamiento en
# cada predicción, y el Stacking lo repite tres veces por su validación interna, así que
# con las 292.131 filas la corrida pasa de las dos horas. Poniendo None acá se entrena
# con todo: es el paso que hay que correr una vez antes de entregar.
MUESTRA_ENTRENAMIENTO = 100_000

# La métrica de decisión, en el formato que espera scikit-learn.
from sklearn.metrics import make_scorer
SCORER = make_scorer(f1_score, pos_label=config.CLASE_MEDIDA)


def cargar():
    """Datos y máscara de train/test, con el split que viene del dataset."""
    df = pd.read_pickle(config.CHECKPOINT_TEXTO)
    X, y = modelo.separar_x_y(df)
    es_train = (~df[variables.COL_SPLIT]).to_numpy()
    return X, y, es_train


def muestrear(X, y, n, semilla=config.SEED):
    """Muestra estratificada: mantiene la proporción 84/16 de las clases."""
    if n >= len(X):
        return X, y
    rng = np.random.default_rng(semilla)
    indices = []
    for clase in np.unique(y):
        de_la_clase = np.where(y == clase)[0]
        cuantos = int(round(n * len(de_la_clase) / len(y)))
        indices.append(rng.choice(de_la_clase, min(cuantos, len(de_la_clase)), replace=False))
    indices = np.sort(np.concatenate(indices))
    return X.iloc[indices], y.iloc[indices]


def medir(estimador, X, y, es_train, etiqueta, hiperparametros=""):
    """Entrena con todo el train, evalúa contra todo el test y arma la fila de la tabla."""
    inicio = time.perf_counter()
    Xtr, ytr = X[es_train], y[es_train]
    if MUESTRA_ENTRENAMIENTO:
        Xtr, ytr = muestrear(Xtr, ytr, MUESTRA_ENTRENAMIENTO)
    estimador.fit(Xtr, ytr)
    f1_train = f1_score(ytr, estimador.predict(Xtr), pos_label=config.CLASE_MEDIDA)
    f1_test = f1_score(y[~es_train], estimador.predict(X[~es_train]),
                       pos_label=config.CLASE_MEDIDA)
    return {
        "Modelo": etiqueta,
        "Hiperparámetros": hiperparametros,
        "Métrica en prueba": round(f1_test, 4),
        "Métrica en train": round(f1_train, 4),
        "Brecha": round(f1_test - f1_train, 4),
        "Tiempo (s)": round(time.perf_counter() - inicio, 1),
    }


# --------------------------------------------------------------------------------------
# Parte A — desbalanceo
# --------------------------------------------------------------------------------------

def tecnicas_de_remuestreo():
    """Las cuatro, en el orden que pide la consigna.

    Todas van dentro de imblearn.pipeline.Pipeline y NO del de sklearn: el de imblearn
    aplica el sampler sólo en `fit`; el de sklearn lo aplicaría también en `predict` y
    remuestrearía el test, que es contaminarlo.

    `class_weight="balanced"` va primero porque no toca los datos: reponderar el error
    es siempre menos invasivo que inventar o tirar filas.
    """
    bosque_sin_peso = dict(config.MODELO_CONGELADO)
    bosque_sin_peso["class_weight"] = None
    return [
        ("class_weight='balanced' (sin tocar los datos)",
         PipelineImb([("modelo", RandomForestClassifier(**config.MODELO_CONGELADO))])),
        ("RandomUnderSampler",
         PipelineImb([("remuestreo", RandomUnderSampler(random_state=config.SEED)),
                      ("modelo", RandomForestClassifier(**bosque_sin_peso))])),
        ("SMOTE",
         PipelineImb([("remuestreo", SMOTE(random_state=config.SEED, k_neighbors=5)),
                      ("modelo", RandomForestClassifier(**bosque_sin_peso))])),
        ("ADASYN",
         PipelineImb([("remuestreo", ADASYN(random_state=config.SEED, n_neighbors=5)),
                      ("modelo", RandomForestClassifier(**bosque_sin_peso))])),
    ]


# --------------------------------------------------------------------------------------
# Parte B — modelos, diversificados por familia
# --------------------------------------------------------------------------------------

def rejillas():
    """Un espacio de búsqueda por modelo base.

    La diversificación es por FAMILIA, no por nombre: RandomForest, ExtraTrees y
    GradientBoosting son los tres bosques de árboles y sus errores están
    correlacionados, así que ensamblarlos no gana nada. Acá hay bagging de árboles,
    boosting, un lineal embolsado y uno por distancias.
    """
    return {
        "RandomForest (bagging de árboles)": (
            RandomForestClassifier(random_state=config.SEED, n_jobs=-1,
                                   class_weight="balanced"),
            {"n_estimators": [200, 300, 500], "max_depth": [8, 12, 16, None],
             "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.3]},
        ),
        "GradientBoosting (boosting)": (
            GradientBoostingClassifier(random_state=config.SEED),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 8],
             "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0]},
        ),
        "Bagging de regresión logística (lineal)": (
            PipelineImb([("escala", MinMaxScaler()),
                         ("modelo", BaggingClassifier(
                             LogisticRegression(max_iter=1000, class_weight="balanced"),
                             random_state=config.SEED, n_jobs=-1))]),
            {"modelo__n_estimators": [10, 25], "modelo__max_samples": [0.5, 1.0]},
        ),
        "KNN (distancias)": (
            # Normalizado entre 0 y 1: sin eso, `actividad_lector` (0 a 2.398) domina
            # la distancia y las variables en [0,1] no pesan nada.
            PipelineImb([("escala", MinMaxScaler()),
                         ("modelo", KNeighborsClassifier(n_jobs=-1))]),
            {"modelo__n_neighbors": [15, 25, 50], "modelo__weights": ["uniform", "distance"]},
        ),
    }


def optimizar(nombre, estimador, rejilla, X, y, n_iter=8):
    """RandomizedSearch sobre la muestra. Devuelve el estimador ya configurado."""
    inicio = time.perf_counter()
    busqueda = RandomizedSearchCV(
        estimador, rejilla, n_iter=n_iter, scoring=SCORER,
        cv=StratifiedKFold(3, shuffle=True, random_state=config.SEED),
        random_state=config.SEED, n_jobs=-1, error_score="raise",
    )
    busqueda.fit(X, y)
    print(f"  {nombre}: f1={busqueda.best_score_:.4f} en {time.perf_counter()-inicio:.0f}s")
    print(f"    {busqueda.best_params_}")
    return busqueda.best_estimator_, busqueda.best_params_


# --------------------------------------------------------------------------------------
# Figuras del informe
# --------------------------------------------------------------------------------------

def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#e3e2de",
                         "grid.color": "#e3e2de", "axes.grid": True, "axes.axisbelow": True,
                         "font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight",
                         "axes.titlelocation": "left", "axes.titleweight": "bold"})
    return plt


def boxplot_de_folds(resultados_cv: dict, nombre="13_boxplot_folds"):
    """Los 10 folds de cada modelo, en un boxplot.

    La cátedra pide este gráfico en lugar de escribir los folds en prosa: la
    dispersión entre folds dice si la diferencia entre dos modelos es real o entra
    dentro de lo que varía un mismo modelo según cómo se parta el dataset.
    """
    plt = _estilo()
    nombres = list(resultados_cv)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    # matplotlib 3.9 renombró `labels` a `tick_labels` en boxplot y 3.11 lo eliminó.
    # Se pasa por nombre nuevo si existe, para no depender de la versión instalada.
    etiquetas = [n[:38] for n in nombres]
    try:
        caja = ax.boxplot([resultados_cv[n] for n in nombres], vert=False,
                          patch_artist=True, tick_labels=etiquetas, widths=0.55)
    except TypeError:
        caja = ax.boxplot([resultados_cv[n] for n in nombres], vert=False,
                          patch_artist=True, labels=etiquetas, widths=0.55)
    for parche in caja["boxes"]:
        parche.set_facecolor("#2a78d6"); parche.set_alpha(0.75); parche.set_edgecolor("#2a78d6")
    for mediana in caja["medians"]:
        mediana.set_color("#fcfcfb"); mediana.set_linewidth(2)
    ax.set_xlabel(f"{config.METRICA} sobre la clase {config.CLASE_MEDIDA}")
    ax.set_title(f"Validación cruzada en {FOLDS} partes")
    ax.grid(axis="y", visible=False)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.png")
    plt.close(fig)
    print(f"  → {nombre}.png")


def _importancias_nativas(estimador):
    """Importancias exactas del modelo, sin aproximaciones.

    Los modelos de árboles traen `feature_importances_`. Un modelo lineal no, pero su
    equivalente exacto es el módulo de sus coeficientes: como las variables entran
    escaladas entre 0 y 1, los coeficientes son directamente comparables entre sí. En
    un bagging se promedia el coeficiente de cada estimador de la bolsa.

    Nada de SHAP: está prohibida por CLAUDE.md 3.1 y además es una aproximación cuando
    el cálculo exacto ya viene con el modelo.
    """
    # Un Pipeline delega en su último paso.
    if hasattr(estimador, "steps"):
        estimador = estimador.steps[-1][1]

    if hasattr(estimador, "feature_importances_"):
        return estimador.feature_importances_

    if hasattr(estimador, "coef_"):
        return np.abs(np.ravel(estimador.coef_))

    # Bagging: el promedio de los coeficientes de la bolsa. Cada estimador ve un
    # subconjunto de columnas, así que se acumula en las posiciones que le tocaron.
    if hasattr(estimador, "estimators_") and hasattr(estimador, "estimators_features_"):
        n_columnas = max(max(f) for f in estimador.estimators_features_) + 1
        acumulado = np.zeros(n_columnas)
        cuenta = np.zeros(n_columnas)
        for sub, columnas_vistas in zip(estimador.estimators_,
                                        estimador.estimators_features_):
            if not hasattr(sub, "coef_"):
                return None
            acumulado[columnas_vistas] += np.abs(np.ravel(sub.coef_))
            cuenta[columnas_vistas] += 1
        return np.divide(acumulado, np.maximum(cuenta, 1))

    return None


def grafico_importancias(estimador, columnas, nombre="14_importancias", cuantas=25):
    """Importancias nativas del mejor modelo. No SHAP: está prohibida y además es una
    aproximación cuando el cálculo exacto ya viene con el modelo."""
    valores = _importancias_nativas(estimador)
    if valores is None:
        print("  (no se pudieron obtener importancias nativas de este modelo)")
        return None
    plt = _estilo()
    importancias = pd.Series(valores, index=columnas)
    top = importancias.sort_values(ascending=False).head(cuantas).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, cuantas * 0.26)))
    ax.barh(top.index, top.values, color="#2a78d6", height=0.72)
    ax.set_xlabel("importancia" if hasattr(estimador, "feature_importances_")
                  else "|coeficiente| promedio de la bolsa")
    ax.set_title(f"Las {cuantas} variables más importantes del modelo elegido")
    ax.grid(axis="y", visible=False)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.png")
    plt.close(fig)
    print(f"  → {nombre}.png")
    return importancias.sort_values(ascending=False)


def matriz_de_confusion(y_real, y_pred, nombre="15_matriz_de_confusion"):
    """Material descriptivo: las decisiones se toman con la métrica, no con esto."""
    plt = _estilo()
    matriz = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matriz, cmap="Blues")
    etiquetas = ["0 · no gustó", "1 · gustó"]
    ax.set_xticks([0, 1], etiquetas); ax.set_yticks([0, 1], etiquetas)
    ax.set_xlabel("predicho"); ax.set_ylabel("real")
    limite = matriz.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]:,}", ha="center", va="center",
                    color="#fcfcfb" if matriz[i, j] > limite else "#0b0b0b",
                    fontweight="bold")
    ax.set_title("Matriz de confusión del modelo elegido")
    ax.grid(False)
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.png")
    plt.close(fig)
    print(f"  → {nombre}.png")
    return matriz


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def main() -> pd.DataFrame:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    X, y, es_train = cargar()
    print(f"train {es_train.sum():,} × {X.shape[1]} · test {(~es_train).sum():,}\n")
    Xb, yb = muestrear(X[es_train], y[es_train], MUESTRA_BUSQUEDA)
    Xcv, ycv = muestrear(X[es_train], y[es_train], MUESTRA_CV, semilla=config.SEED + 1)
    entrena = MUESTRA_ENTRENAMIENTO or int(es_train.sum())
    print(f"búsqueda {len(Xb):,} · entrenamiento final {entrena:,} · "
          f"validación cruzada {len(Xcv):,} · TEST COMPLETO {(~es_train).sum():,}\n")

    # ---------- Parte A ----------
    print("=" * 100); print("PARTE A — DESBALANCEO"); print("=" * 100)
    filas_a = []
    for etiqueta, tuberia in tecnicas_de_remuestreo():
        try:
            fila = medir(tuberia, X, y, es_train, etiqueta)
        except Exception as error:                      # SMOTE/ADASYN pueden no escalar
            print(f"  {etiqueta}: falló ({type(error).__name__}: {error})")
            continue
        filas_a.append(fila)
        print(f"  {etiqueta:48s} f1={fila['Métrica en prueba']:.4f} "
              f"brecha={fila['Brecha']:+.4f} ({fila['Tiempo (s)']}s)")
    tabla_a = pd.DataFrame(filas_a)
    tabla_a.to_csv(config.DIR_RESULTADOS / "07_desbalanceo.csv", index=False)

    # ---------- Parte B ----------
    print("\n" + "=" * 100); print("PARTE B — MODELOS"); print("=" * 100)
    print("Optimización de hiperparámetros (RandomizedSearch sobre la muestra):")
    optimizados, parametros = {}, {}
    for nombre, (estimador, rejilla) in rejillas().items():
        optimizados[nombre], parametros[nombre] = optimizar(nombre, estimador, rejilla, Xb, yb)

    # Los ensambles se arman DESPUÉS, con los modelos base ya optimizados: el stacking
    # no optimiza nada por su cuenta, sólo aprende a combinar lo que le den.
    base = [("rf", optimizados["RandomForest (bagging de árboles)"]),
            ("gb", optimizados["GradientBoosting (boosting)"]),
            ("knn", optimizados["KNN (distancias)"])]
    optimizados["Voting (RF + GB + KNN)"] = VotingClassifier(base, voting="soft", n_jobs=-1)
    optimizados["Stacking (RF + GB + KNN → logística)"] = StackingClassifier(
        base, final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=3, n_jobs=-1)

    print("\nEntrenamiento final sobre TODO el train y evaluación contra TODO el test:")
    filas_b = []
    for nombre, estimador in optimizados.items():
        fila = medir(estimador, X, y, es_train, nombre,
                     str(parametros.get(nombre, "combinación de los base optimizados")))
        filas_b.append(fila)
        print(f"  {nombre:42s} f1={fila['Métrica en prueba']:.4f} "
              f"brecha={fila['Brecha']:+.4f} ({fila['Tiempo (s)']}s)")

    # ---------- Selección por validación cruzada ----------
    print(f"\nValidación cruzada en {FOLDS} partes sobre la muestra:")
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=config.SEED)
    resultados_cv = {}
    for nombre, estimador in optimizados.items():
        inicio = time.perf_counter()
        puntajes = cross_val_score(estimador, Xcv, ycv, scoring=SCORER, cv=cv, n_jobs=-1)
        resultados_cv[nombre] = puntajes
        print(f"  {nombre:42s} {puntajes.mean():.4f} ± {puntajes.std():.4f} "
              f"({time.perf_counter()-inicio:.0f}s)")

    tabla_b = pd.DataFrame(filas_b)
    tabla_b["CV media"] = [round(resultados_cv[n].mean(), 4) for n in tabla_b["Modelo"]]
    tabla_b["CV desvío"] = [round(resultados_cv[n].std(), 4) for n in tabla_b["Modelo"]]
    referencia = tabla_b["Métrica en prueba"].max()
    tabla_b["Diferencia"] = (tabla_b["Métrica en prueba"] - referencia).round(4)

    # El criterio: primero se descarta por brecha, después se elige por métrica.
    UMBRAL_BRECHA = 0.05
    aceptables = tabla_b[tabla_b["Brecha"].abs() <= UMBRAL_BRECHA]
    if aceptables.empty:
        print(f"\n(ningún modelo baja de {UMBRAL_BRECHA} de brecha; se elige por métrica)")
        aceptables = tabla_b
    ganador = aceptables.loc[aceptables["Métrica en prueba"].idxmax(), "Modelo"]
    tabla_b["Elegido"] = np.where(tabla_b["Modelo"] == ganador, "✓", "")

    columnas = ["Modelo", "Hiperparámetros", "Métrica en prueba", "Diferencia", "Brecha",
                "CV media", "CV desvío", "Tiempo (s)", "Elegido"]
    tabla_b[columnas].to_csv(config.DIR_RESULTADOS / "08_modelos.csv", index=False)

    print("\n" + "=" * 100)
    print(f"MODELO ELEGIDO: {ganador}")
    print("(criterio: primero se descarta por brecha > 0.05, después se elige por métrica)")
    print("=" * 100)
    print(tabla_b[columnas].to_string(index=False))

    # ---------- Cierre ----------
    # El estado se guarda ANTES de dibujar: una corrida de 50 minutos no puede
    # perderse porque falle una llamada a matplotlib.
    import pickle
    with open(config.DIR_CHECKPOINTS / "07_corrida_modelos.pkl", "wb") as archivo:
        pickle.dump({"tabla": tabla_b, "cv": resultados_cv, "ganador": ganador,
                     "modelos": optimizados}, archivo)
    print(f"\nEstado guardado: {config.DIR_CHECKPOINTS / '07_corrida_modelos.pkl'}")

    print("\nFiguras:")
    boxplot_de_folds(resultados_cv)
    elegido = optimizados[ganador]
    Xtr, ytr = X[es_train], y[es_train]
    if MUESTRA_ENTRENAMIENTO:
        Xtr, ytr = muestrear(Xtr, ytr, MUESTRA_ENTRENAMIENTO)
    elegido.fit(Xtr, ytr)
    predicho = elegido.predict(X[~es_train])
    grafico_importancias(elegido, X.columns)
    matriz_de_confusion(y[~es_train], predicho)

    with open(config.DIR_CHECKPOINTS / "07_modelo_final.pkl", "wb") as archivo:
        pickle.dump({"nombre": ganador, "estimador": elegido,
                     "columnas": list(X.columns)}, archivo)
    print(f"\nGuardado: {config.DIR_CHECKPOINTS / '07_modelo_final.pkl'}")
    return tabla_b


if __name__ == "__main__":
    main()
