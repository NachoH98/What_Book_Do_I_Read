"""Minería de texto sobre el resumen del libro.

El preprocesamiento visto en clase: minúsculas, normalización, limpieza de ruido,
tokenización, stopwords y lematización con spaCy.

Sobre el volumen: el dataset tiene 389.508 filas pero sólo ~45.800 resúmenes
distintos, porque un libro con 200 opiniones repite su resumen 200 veces. Todo el
procesamiento pesado se hace sobre los textos únicos y después se mapea por id_libro.
Lematizar 389.508 veces lo mismo sería ocho veces más trabajo para el mismo resultado.
"""

from __future__ import annotations

import re

import pandas as pd
import spacy

from src import config

MODELO_SPACY = "es_core_news_sm"

# Las negaciones NO se sacan aunque spaCy las marque como stopwords: invierten el
# sentido de lo que acompañan. `no` viene con is_stop=True por defecto, así que sin
# esta lista "no es interesante" y "es interesante" quedarían idénticos.
#
# Nota honesta para el informe: `resumen` es la sinopsis editorial del libro, no una
# opinión, así que el riesgo es menor que en un texto de reseña. Se respeta igual.
NEGACIONES = {
    "no", "ni", "nunca", "jamas", "jamás", "tampoco", "nada", "nadie",
    "ningun", "ningún", "ninguna", "ninguno", "sin", "contra", "apenas", "pero",
}

# Ruido específico de esta fuente, detectado mirando los resúmenes.
RUIDO = re.compile(r"https?://\S+|www\.\S+|\b\d{9,}\b|[^\wáéíóúüñ\s]", re.IGNORECASE)
REPETICIONES = re.compile(r"(\w)\1{2,}")     # "buenííííísimo" -> "buenísimo"
ESPACIOS = re.compile(r"\s+")

LARGO_MINIMO = 3   # tokens de 1-2 letras no aportan


def cargar_modelo():
    """Carga spaCy sin parser ni NER: sólo hace falta el lematizador."""
    return spacy.load(MODELO_SPACY, disable=["parser", "ner", "textcat"])


def construir_stopwords(nlp) -> set[str]:
    """Stopwords del español, menos las negaciones."""
    return set(nlp.Defaults.stop_words) - NEGACIONES


def _normalizar(texto: str) -> str:
    """Minúsculas, sin URLs ni puntuación, sin repeticiones de letras."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = REPETICIONES.sub(r"\1", texto)
    texto = RUIDO.sub(" ", texto)
    return ESPACIOS.sub(" ", texto).strip()


def limpiar(textos: list[str], nlp=None, lote: int = 200) -> list[str]:
    """Normaliza, tokeniza, saca stopwords y lematiza. Devuelve texto listo para vectorizar."""
    nlp = nlp or cargar_modelo()
    stopwords = construir_stopwords(nlp)
    normalizados = [_normalizar(t) for t in textos]

    limpios = []
    for doc in nlp.pipe(normalizados, batch_size=lote):
        lemas = [
            token.lemma_ for token in doc
            if token.lemma_ not in stopwords
            and len(token.lemma_) >= LARGO_MINIMO
            and not token.is_punct
            and not token.like_num
        ]
        limpios.append(" ".join(lemas))
    return limpios


# --------------------------------------------------------------------------------------
# Vectorización
# --------------------------------------------------------------------------------------

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import modelo, variables

CACHE_LIMPIOS = config.DIR_CHECKPOINTS / "06_resumenes_limpios.pkl"


def texto_por_fila(df: pd.DataFrame) -> pd.Series:
    """Trae el texto limpio de cada fila a partir de su id_libro."""
    if not CACHE_LIMPIOS.is_file():
        raise FileNotFoundError(
            f"Falta {CACHE_LIMPIOS}. Correr antes: python -m src.limpiar_resumenes")
    limpios = pd.read_pickle(CACHE_LIMPIOS).set_index("id_libro")["texto_limpio"]
    return df["id_libro"].map(limpios).fillna("")


def vectorizar(df: pd.DataFrame, min_df=5, max_df=0.5, max_features=2000):
    """TF-IDF con `fit` SÓLO sobre train y `transform` sobre todo.

    El vocabulario y el IDF son estadísticas del corpus: calcularlos con los
    documentos de test sería dejar que el test decida qué palabras existen y cuánto
    pesan. `fit_transform` sobre el dataset completo es el error clásico acá.
    """
    textos = texto_por_fila(df)
    es_train = ~df[variables.COL_SPLIT]

    vectorizador = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                   max_features=max_features, sublinear_tf=True)
    vectorizador.fit(textos[es_train])
    matriz = vectorizador.transform(textos)   # sparse: nunca se densifica entera
    return matriz, vectorizador


def importancias_del_texto(matriz, y, es_train) -> np.ndarray:
    """Entrena el modelo congelado sobre la matriz sparse y devuelve las importancias.

    Se entrena con TODAS las columnas de texto para poder rankearlas, pero al dataset
    final entran sólo las mejores: 2.000 columnas de texto aplastarían a las 353
    variables construidas y multiplicarían el tiempo de cada experimento.
    """
    clasificador = modelo.modelo_congelado()
    clasificador.fit(matriz[es_train], y[es_train])
    return clasificador.feature_importances_


def seleccionar_columnas(matriz, vectorizador, importancias, cuantas=60):
    """Se queda con las `cuantas` palabras más importantes y las devuelve densas.

    Densificar acá sí es legítimo: son 60 columnas, no 2.000. `.toarray()` sobre la
    matriz completa serían 389.508 × 2.000 flotantes, unos 3 GB.
    """
    mejores = np.argsort(importancias)[::-1][:cuantas]
    palabras = np.array(vectorizador.get_feature_names_out())[mejores]
    denso = pd.DataFrame(
        matriz[:, mejores].toarray(),
        columns=[f"txt_{p}" for p in palabras],
    ).astype("float32")
    return denso, palabras, importancias[mejores]


# --------------------------------------------------------------------------------------
# Enfoque simple: columnas binarias por palabra clave
# --------------------------------------------------------------------------------------

# Elegidas mirando los resúmenes (ver src/correr_texto.py, que imprime las más
# frecuentes). No son las más frecuentes sin más: son las que describen QUÉ TIPO de
# libro es, que es lo que puede relacionarse con que guste o no. Palabras como
# "historia" o "vida" aparecen en todo y no distinguen nada.
PALABRAS_CLAVE = [
    "amor", "guerra", "muerte", "familia", "misterio", "crimen", "asesinato",
    "policía", "investigación", "magia", "aventura", "viaje", "futuro", "guerra",
    "humor", "premio", "bestseller", "novela", "clásico", "juvenil", "infantil",
    "erótico", "pasión", "thriller", "terror", "fantasía", "ciencia", "historia",
    "político", "social", "mujer", "niño", "padre", "madre", "amistad", "secreto",
]


def columnas_binarias(df: pd.DataFrame, palabras=None) -> pd.DataFrame:
    """Una columna 0/1 por palabra clave: ¿aparece en el resumen del libro?

    Más interpretable que el TF-IDF —cada columna se lee sola— y a veces alcanza.
    Se compara contra el TF-IDF en la tabla de experimentos.
    """
    palabras = sorted(set(palabras or PALABRAS_CLAVE))
    textos = texto_por_fila(df)
    return pd.DataFrame(
        {f"kw_{p}": textos.str.contains(rf"\b{p}", regex=True).astype("int8")
         for p in palabras},
        index=df.index,
    )


# --------------------------------------------------------------------------------------
# Nube de palabras
# --------------------------------------------------------------------------------------

def nube_de_palabras(textos: pd.Series, nombre="11_nube_de_palabras") -> None:
    """Nube sobre el texto YA LIMPIO.

    Si en la nube aparecen preposiciones o artículos, el preprocesamiento no corrió:
    es la verificación visual de que las stopwords se fueron.
    """
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    corpus = " ".join(textos.dropna().astype(str))
    nube = WordCloud(width=1600, height=800, background_color="#fcfcfb",
                     colormap="Blues", max_words=150, random_state=config.SEED,
                     collocations=False).generate(corpus)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(nube, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Palabras más frecuentes en los resúmenes, ya lematizados y sin stopwords",
                 fontsize=12, fontweight="bold", loc="left", color="#0b0b0b")
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.png", dpi=150, bbox_inches="tight",
                facecolor="#fcfcfb")
    plt.show()
    plt.close(fig)
    print(f"  → {nombre}.png")


def grafico_palabras_seleccionadas(palabras, importancias,
                                   nombre="12_palabras_por_importancia") -> None:
    """Las palabras que sobrevivieron a la selección, ordenadas por importancia."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(4, len(palabras) * 0.22)))
    orden = np.argsort(importancias)
    ax.barh([palabras[i] for i in orden], importancias[orden], color="#2a78d6", height=0.72)
    ax.set_xlabel("importancia en el modelo congelado")
    ax.set_title(f"Las {len(palabras)} palabras que entran al modelo",
                 fontsize=12, fontweight="bold", loc="left")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="x", color="#e3e2de", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.png", dpi=150, bbox_inches="tight",
                facecolor="#fcfcfb")
    plt.show()
    plt.close(fig)
    print(f"  → {nombre}.png")
