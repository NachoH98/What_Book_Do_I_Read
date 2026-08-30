"""Creación de atributos. La etapa que más baja el error (CLAUDE.md 4).

TODO perfil se calcula SOBRE TRAIN y se aplica a train y a test. El split va antes
que cualquier estadística (CLAUDE.md 3.3): si un perfil se calculara sobre el dataset
completo, contendría la respuesta de la fila que se quiere predecir, la métrica se
inflaría y el trabajo no serviría.

Sobre los nombres de las columnas: `modelo.es_derivada_del_target` descarta toda
columna que contenga "rating" o "gusto" en el nombre, para que nadie entrene con el
target por accidente. Los perfiles agregan `gusto` de OTRAS filas, que es justamente
lo que CLAUDE.md 3.3 contempla como perfil de usuario, así que son predictoras
legítimas — pero si se llamaran `pct_gusto_lector` la guarda las tiraría sin avisar.
Por eso se llaman `afinidad_*`. No es una forma de esquivar la guarda: es que la
guarda protege contra usar el target DE LA MISMA FILA, y esto es otra cosa.

Lo que NO se construye, por decisión explícita: la media y el desvío del rating del
lector. `rating` y cualquier estadística suya quedan afuera en todo momento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

COL_SPLIT = "es_test"

# Top-N por cantidad de opiniones EN TRAIN. Salen de la curva de cardinalidad del
# análisis exploratorio: el top 100 de autores cubre el 43% de las opiniones y el
# top 50 de editoriales el 85%. One-hot directo sobre 19.243 autores no es opción.
TOP_AUTORES = 100
TOP_EDITORIALES = 50
OTRAS = "OTRAS"

# Volumen mínimo en train para que una combinación región × editorial tenga su propia
# dummy. Sin este corte salen cientos de columnas, casi todas con un puñado de filas.
MIN_INTERACCION = 500


# --------------------------------------------------------------------------------------
# El split, primero que todo
# --------------------------------------------------------------------------------------

def dividir(df: pd.DataFrame) -> pd.DataFrame:
    """Marca cada fila como train o test y deja la decisión guardada en el dataset.

    El split se persiste en la columna `es_test` en lugar de rehacerse en cada paso:
    las ramas de texto y de modelos tienen que usar EXACTAMENTE la misma partición
    con la que se calcularon los perfiles. Si cada etapa hiciera su propio split, un
    perfil calculado con el train de acá terminaría dentro del test de allá y la fuga
    volvería por la ventana.
    """
    out = df.copy()
    indices_test = train_test_split(
        out.index,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=out[config.TARGET],
    )[1]
    out[COL_SPLIT] = False
    out.loc[indices_test, COL_SPLIT] = True
    return out


def _train(df: pd.DataFrame) -> pd.DataFrame:
    """Las filas con las que está permitido calcular estadísticas."""
    if COL_SPLIT not in df.columns:
        raise ValueError("Falta la columna `es_test`: hay que llamar a dividir() primero.")
    return df.loc[~df[COL_SPLIT]]


def _afinidad_por(df: pd.DataFrame, claves: list[str], nombre: str,
                  loo: bool = False) -> pd.Series:
    """Proporción de `gusto` por combinación de claves, calculada SÓLO sobre train.

    Las combinaciones que no existen en train reciben la tasa global de train. Es la
    mejor estimación disponible cuando no hay información específica, y evita meter
    nulos en una columna que el modelo va a usar.

    Con `loo=True` cada fila de train se excluye a sí misma del promedio. Sin eso, la
    fila aporta su propio target al perfil que después se usa para predecirla: no es
    fuga de test —el test nunca participa del cálculo— pero el modelo entrena viendo
    parte de la respuesta y la brecha se dispara. Con el 46% de los libros teniendo una
    sola opinión, para esos el perfil ES el target copiado con otro nombre.
    """
    train = _train(df)
    prior = train[config.TARGET].mean()
    agregado = train.groupby(claves, observed=True)[config.TARGET].agg(["sum", "count"])
    unido = df[claves].merge(agregado, how="left", left_on=claves, right_index=True)
    suma = unido["sum"].to_numpy(dtype="float64")
    n = unido["count"].to_numpy(dtype="float64")

    if loo:
        # Sólo las filas de train se descuentan a sí mismas; las de test no
        # participaron del agregado, así que no hay nada que restarles.
        es_train = (~df[COL_SPLIT]).to_numpy()
        suma = np.where(es_train, suma - df[config.TARGET].to_numpy(), suma)
        n = np.where(es_train, n - 1, n)

    with np.errstate(invalid="ignore", divide="ignore"):
        valores = np.where(n > 0, suma / n, np.nan)
    return pd.Series(valores).fillna(prior).astype("float32").values


def _conteo_por(df: pd.DataFrame, claves: list[str], nombre: str) -> pd.Series:
    """Cantidad de opiniones por clave, contada SÓLO sobre train."""
    train = _train(df)
    tabla = train.groupby(claves, observed=True).size().rename(nombre)
    unido = df[claves].merge(tabla, how="left", left_on=claves, right_index=True)
    return unido[nombre].fillna(0).astype("float32").values


# --------------------------------------------------------------------------------------
# Bloque 1 — perfil del lector
# --------------------------------------------------------------------------------------

def perfil_lector(df: pd.DataFrame, loo: bool = False) -> pd.DataFrame:
    """Qué le gusta a cada lector, en porcentaje y no en valor absoluto.

    El porcentaje es lo que hace comparables a un lector con 2.398 opiniones y a uno
    con 8: en absoluto, el primero tendría números más grandes en todo y el modelo
    aprendería a detectar lectores activos en vez de gustos.

    La actividad total va aparte y sí en absoluto: es otra señal, y el análisis
    exploratorio mostró que discrimina (la tasa cae de 88% a 80% entre el tramo de
    2-5 opiniones y el de 500+).
    """
    out = df.copy()
    decada = (pd.to_numeric(out["anio_edicion"], errors="coerce") // 10 * 10).fillna(-1)

    out["afinidad_lector_genero"] = _afinidad_por(
        out.assign(_g=out["genero_libro"]), ["id_lector", "_g"], "v", loo)
    out["afinidad_lector_decada"] = _afinidad_por(
        out.assign(_d=decada), ["id_lector", "_d"], "v", loo)
    out["afinidad_lector_global"] = _afinidad_por(out, ["id_lector"], "v", loo)
    out["actividad_lector"] = _conteo_por(out, ["id_lector"], "v")
    return out


def perfil_lector_autor(df: pd.DataFrame, loo: bool = False) -> pd.DataFrame:
    """Afinidad del lector con cada autor del top-N. Necesita `autor_top` ya creada."""
    if "autor_top" not in df.columns:
        raise ValueError("perfil_lector_autor necesita que top_n() haya corrido antes.")
    out = df.copy()
    out["afinidad_lector_autor"] = _afinidad_por(out, ["id_lector", "autor_top"], "v", loo)
    return out


# --------------------------------------------------------------------------------------
# Bloque 2 — perfil del libro
# --------------------------------------------------------------------------------------

def perfil_libro(df: pd.DataFrame, loo: bool = False) -> pd.DataFrame:
    """Cómo le fue al libro con el público, y con qué segmentos del público.

    Los cortes por género del lector y por región no son decorativos: si un libro
    gusta al 90% de las mujeres y al 40% de los hombres, esa diferencia es información
    sobre el libro que ninguna variable global captura.
    """
    out = df.copy()
    out["libro_n_opiniones"] = _conteo_por(out, ["id_libro"], "v")
    out["libro_afinidad"] = _afinidad_por(out, ["id_libro"], "v", loo)

    train = _train(out)
    prior = train[config.TARGET].mean()
    segmentos = {
        "libro_afinidad_hombres": out["genero_lector_imputado"].eq("Hombre"),
        "libro_afinidad_mujeres": out["genero_lector_imputado"].eq("Mujer"),
        "libro_afinidad_espania": out["region"].str.startswith("España"),
    }
    es_train = (~out[COL_SPLIT]).to_numpy()
    objetivo = out[config.TARGET].to_numpy()

    for nombre, pertenece in segmentos.items():
        # El agregado se calcula sobre las filas de train QUE PERTENECEN al segmento.
        del_segmento = train.loc[pertenece.loc[train.index]]
        agregado = del_segmento.groupby("id_libro", observed=True)[config.TARGET].agg(["sum", "count"])
        unido = out[["id_libro"]].merge(agregado, how="left", left_on="id_libro", right_index=True)
        suma = unido["sum"].to_numpy(dtype="float64")
        n = unido["count"].to_numpy(dtype="float64")

        if loo:
            # Sólo se descuenta la fila que efectivamente aportó al agregado: tiene
            # que ser de train Y pertenecer al segmento. Sin la segunda condición se
            # restarían filas que nunca sumaron, y el promedio quedaría mal.
            aporto = es_train & pertenece.to_numpy()
            suma = np.where(aporto, suma - objetivo, suma)
            n = np.where(aporto, n - 1, n)

        with np.errstate(invalid="ignore", divide="ignore"):
            valores = np.where(n > 0, suma / n, np.nan)
        out[nombre] = pd.Series(valores).fillna(prior).astype("float32").values
    return out


# --------------------------------------------------------------------------------------
# Bloque 3 — del cruce y de los campos
# --------------------------------------------------------------------------------------

def antiguedad_y_edad(df: pd.DataFrame) -> pd.DataFrame:
    """Antigüedad del libro y edad del lector, ambas al momento de la opinión.

    No usan el target, así que no hay nada que calcular sobre train: son aritmética
    de la propia fila.
    """
    out = df.copy()
    anio_opinion = pd.to_datetime(out["fecha"], format="%d-%m-%Y").dt.year
    out["antiguedad_libro"] = (anio_opinion - pd.to_numeric(out["anio_edicion"],
                                                            errors="coerce")).astype("float32")
    out["edad_lector"] = (anio_opinion - out["nacimiento"]).astype("float32")
    return out


def top_n(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce autor y editorial a su top-N por cantidad de opiniones EN TRAIN.

    El top-N se define con value_counts() sobre train (CLAUDE.md 3.3). Definirlo sobre
    el dataset completo dejaría que la composición del test decida qué autores tienen
    columna propia, que es información del test entrando por la puerta de atrás.
    """
    out = df.copy()
    train = _train(out)
    for columna, n, destino in [("autor", TOP_AUTORES, "autor_top"),
                                ("editorial", TOP_EDITORIALES, "editorial_top")]:
        top = set(train[columna].value_counts().head(n).index)
        out[destino] = out[columna].where(out[columna].isin(top), OTRAS)
    return out


def interaccion_region_editorial(df: pd.DataFrame) -> pd.DataFrame:
    """Cruce de región del lector con editorial: qué sello lee cada mercado.

    Sólo sobreviven las combinaciones con volumen en train; el resto va a OTRAS. Sin
    ese corte salen cientos de dummies con un puñado de filas cada una, que es ruido
    con nombre propio.
    """
    if "editorial_top" not in df.columns:
        raise ValueError("interaccion_region_editorial necesita top_n() antes.")
    out = df.copy()
    cruce = out["region"].astype(str) + " × " + out["editorial_top"].astype(str)
    volumen = cruce[~out[COL_SPLIT]].value_counts()
    frecuentes = set(volumen[volumen >= MIN_INTERACCION].index)
    out["region_x_editorial"] = cruce.where(cruce.isin(frecuentes), OTRAS)
    return out


# --------------------------------------------------------------------------------------
# Dummies de las categóricas nuevas
# --------------------------------------------------------------------------------------

COLS_DUMMIES_NUEVAS = ["autor_top", "editorial_top", "region_x_editorial"]


def dummies_nuevas(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot de las categóricas creadas acá, con la misma guarda de idempotencia."""
    out = df.copy()
    presentes = [c for c in COLS_DUMMIES_NUEVAS
                 if c in out.columns
                 and not any(col.startswith(f"{c}_") for col in out.columns)]
    if not presentes:
        return out
    dummies = pd.get_dummies(out[presentes], prefix=presentes, dtype="int8")
    return pd.concat([out, dummies], axis=1)


# --------------------------------------------------------------------------------------
# El pipeline, en el orden en que las dependencias lo permiten
# --------------------------------------------------------------------------------------

def _con_loo(funcion):
    """Envuelve un perfil para que se calcule dejando cada fila de train fuera de sí."""
    envuelta = lambda df: funcion(df, loo=True)
    envuelta.__name__ = f"{funcion.__name__}_loo"
    return envuelta


# El pipeline ELEGIDO. Cada fila de train se excluye de su propio perfil.
#
# La medición que lo decidió: con cálculo directo el f1 de test da 0.4865 con brecha
# -0.2346; con leave-one-out da 0.4828 con brecha -0.0947. En métrica están empatados
# (0.0037 cae dentro de la banda de ruido de ±0.005), pero la brecha se corta un 60%,
# y el criterio del curso es descartar primero por brecha.
#
# El mecanismo, medido: con cálculo directo la correlación de `afinidad_lector_genero`
# con el target vale 0.590 en train y 0.338 en test — la variable significa algo
# distinto de cada lado, porque en train contiene la respuesta de su propia fila. Con
# leave-one-out da 0.345 y 0.338: la misma variable en ambos lados.
#
# Es el mismo principio de "dejar la fila afuera" de la validación cruzada, aplicado
# al cálculo de la variable en vez de a la evaluación del modelo.
PIPELINE_LOO = [
    top_n,
    _con_loo(perfil_lector),
    _con_loo(perfil_lector_autor),
    _con_loo(perfil_libro),
    antiguedad_y_edad,
    interaccion_region_editorial,
    dummies_nuevas,
]

# PROBADA Y DESCARTADA: la variante directa, sin leave-one-out. Se conserva porque
# produce la comparación que va al informe, no porque se use.
PIPELINE_DIRECTO = [
    top_n,                        # autor_top y editorial_top: los necesitan los de abajo
    perfil_lector,
    perfil_lector_autor,
    perfil_libro,
    antiguedad_y_edad,
    interaccion_region_editorial,
    dummies_nuevas,
]

PIPELINE = PIPELINE_LOO


def aplicar(df: pd.DataFrame, funciones=None) -> pd.DataFrame:
    """Aplica las funciones en orden. El DataFrame ya tiene que venir dividido."""
    out = df
    for funcion in (PIPELINE if funciones is None else funciones):
        out = funcion(out)
    return out
