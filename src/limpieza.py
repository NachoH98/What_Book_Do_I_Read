"""Limpieza. Una función por transformación, todas activables por separado.

Cada función recibe un DataFrame y devuelve uno nuevo, sin mutar el original y sin
depender de estado global. Así el runner puede aplicarlas de a una y medir su efecto
aislado (CLAUDE.md 3.4).

El orden del pipeline es el de CLAUDE.md 3.5: filtrar → outliers a nulo → imputar →
dummies. No es cosmético: imputar antes de mandar los outliers a nulo dejaría los
valores imposibles adentro, y hacer dummies antes de imputar crearía una columna por
cada valor sucio.

Nota sobre el alcance: acá sólo se limpia lo que llegó al dataset a través de las
opiniones. Los libros y lectores sin ninguna opinión ya desaparecieron en el merge y
sus nulos no son un problema nuestro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.genero_por_nombre import GENERO_POR_NOMBRE

DESCONOCIDO = "Desconocido"

# --------------------------------------------------------------------------------------
# Rangos, justificados con la distribución (ver src/correr_limpieza.py)
# --------------------------------------------------------------------------------------

# Un lector no opina antes de los 10 ni después de los 90. Los percentiles 0,5 y 99,5
# de la edad al opinar son 7 y 85, así que el rango deja afuera menos del 1% y no toca
# la masa central. El 1910 se trata aparte: no es un extremo de la distribución, es el
# valor mínimo del selector del formulario, y por eso se descarta aunque caiga dentro.
EDAD_MIN, EDAD_MAX = 10, 90
NACIMIENTO_CENTINELA = 1910

# La imprenta de tipos móviles es de mediados del siglo XV; el tope es el año más
# reciente que aparece en las opiniones. Fuera de ahí el año no es un dato, es basura.
EDICION_MIN = 1450

COL_PAIS = "region"
COL_GENERO = "genero_lector_imputado"


# --------------------------------------------------------------------------------------
# 1. Filtrar
# --------------------------------------------------------------------------------------

def filtrar_sin_target(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta las opiniones sin rating: sin target no sirven ni para entrenar ni para medir."""
    return df[df[config.COL_RATING].notna()].copy()


def descartar_no_libros(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta las opiniones cuyo `id_libro` no existe como libro en el catálogo.

    Se buscó un volumen relevante de revistas, agendas y cuadernos mal catalogados y
    no aparece: los registros problemáticos no son "otro tipo de cosa", son
    referencias a libros que no están en la tabla. Sin título, autor, género ni
    editorial, la fila no aporta ninguna predictora.

    Equivale a haber hecho el merge con `how="inner"`, pero acá queda medido en vez
    de decidido de antemano.
    """
    return df[df["titulo"].notna()].copy()


# --------------------------------------------------------------------------------------
# 2. Outliers a nulo (todavía no se imputa)
# --------------------------------------------------------------------------------------

def anio_nacimiento_a_nulo(df: pd.DataFrame) -> pd.DataFrame:
    """Manda a NaN los años de nacimiento imposibles. No se imputan: el año no se estima.

    La decisión se toma por lector, no por fila: el año de nacimiento es una propiedad
    de la persona, así que si su edad implícita es imposible en alguna de sus opiniones,
    el año declarado no es creíble en ninguna. Anularlo sólo en algunas filas dejaría
    al mismo lector con dos años de nacimiento distintos.
    """
    out = df.copy()
    anio_opinion = pd.to_datetime(out["fecha"], format="%d-%m-%Y").dt.year
    edad = anio_opinion - out["nacimiento"]

    fuera_de_rango = (edad < EDAD_MIN) | (edad > EDAD_MAX)
    lectores_sospechosos = set(out.loc[fuera_de_rango, "id_lector"].unique())

    invalido = (
        out["nacimiento"].eq(NACIMIENTO_CENTINELA)
        | out["id_lector"].isin(lectores_sospechosos)
    )
    out.loc[invalido, "nacimiento"] = np.nan
    return out


def anio_edicion_a_nulo(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte `anio_edicion` a número y manda a NaN lo ilegible y lo imposible.

    La conversión es el grueso del trabajo: la columna viene como texto y hay 833
    opiniones con restos de campos corridos (' (200', ' crít', ' 01-0'). Mientras sea
    texto, el modelo no la puede usar; convertida, pasa a ser una predictora.
    """
    out = df.copy()
    anio = pd.to_numeric(out["anio_edicion"], errors="coerce")
    tope = pd.to_datetime(out["fecha"], format="%d-%m-%Y").dt.year.max()
    out["anio_edicion"] = anio.where(anio.between(EDICION_MIN, tope))
    return out


def edicion_posterior_a_nulo(df: pd.DataFrame) -> pd.DataFrame:
    """Manda a NaN el año de edición cuando es posterior a la fecha de la opinión.

    Son 28.825 opiniones (7,4%). No es un error de carga: el sitio guarda la última
    edición de su catálogo, no el ejemplar que leyó el lector. Como dato del libro que
    se leyó, entonces, es incorrecto en esas filas.

    Se prueba por separado del resto porque anular el 7,4% de una predictora es una
    decisión distinta a corregir basura: hay que medir si compensa.
    """
    out = df.copy()
    anio = pd.to_numeric(out["anio_edicion"], errors="coerce")
    anio_opinion = pd.to_datetime(out["fecha"], format="%d-%m-%Y").dt.year
    out["anio_edicion"] = anio.where(anio <= anio_opinion)
    return out


# --------------------------------------------------------------------------------------
# 3. Imputar
# --------------------------------------------------------------------------------------

# Ciudades de las tres comunidades con lengua cooficial. La agrupación existe porque
# hay editoriales que publican sólo en catalán, gallego o euskera, así que vivir ahí
# cambia la oferta de libros disponible.
CIUDADES_COOFICIALES = {
    # Cataluña
    "barcelona", "tarragona", "girona", "gerona", "lleida", "lerida", "sabadell",
    "badalona", "terrassa", "tarrasa", "hospitalet de llobregat", "l'hospitalet de llobregat",
    "mataro", "mataró", "reus", "manresa", "sant cugat del valles", "vic", "igualada",
    "granollers", "vilanova i la geltru", "cornella de llobregat", "sitges",
    # Galicia
    "a coruña", "la coruña", "vigo", "ourense", "orense", "pontevedra", "lugo",
    "santiago de compostela", "ferrol", "vilagarcia de arousa", "narón", "naron",
    # País Vasco y Navarra euskaldun
    "bilbao", "getxo", "san sebastian", "san sebastián", "donostia", "vitoria",
    "vitoria-gasteiz", "gasteiz", "barakaldo", "irun", "irún", "portugalete",
    "santurtzi", "basauri", "getaria", "eibar", "durango", "leioa", "erandio",
}

PAISES_SUDAMERICA = {
    "chile", "colombia", "peru", "perú", "uruguay", "venezuela", "ecuador", "bolivia",
    "paraguay", "brasil", "brazil", "guyana", "suriname",
}

NULOS_DISFRAZADOS = {"", "-", "--", "¿?", "?", "n/a", "na", ".", "sin especificar"}


def _partir_vive_en(valor) -> tuple[str | None, str | None]:
    """Separa 'ciudad - país'. Devuelve (ciudad, país), cualquiera puede faltar."""
    if pd.isna(valor):
        return None, None
    texto = str(valor).strip().strip("-").strip()
    if texto.lower() in NULOS_DISFRAZADOS:
        return None, None
    if " - " in texto:
        ciudad, _, pais = texto.partition(" - ")
        return ciudad.strip().lower() or None, pais.strip().lower() or None
    # Sin separador el valor es un país suelto ("España", "Argentina") o una ciudad
    # a la que le quedó colgando el guion ("Tenerife -", que ya vino sin país).
    return None, texto.lower() or None


def normalizar_vive_en(df: pd.DataFrame) -> pd.DataFrame:
    """Separa ciudad y país, y los agrupa en regiones con sentido para el problema.

    Los grupos no son geográficos sino de mercado editorial: lo que cambia la oferta
    de libros es la lengua de publicación y el catálogo local.

    `España (sin especificar)` es un grupo propio y no se reparte entre los otros dos:
    son 100.418 opiniones, el 26% del dataset, que dicen sólo "España". Meterlas en
    "castellanohablante" sería inventar que ninguna es de Barcelona o Bilbao; meterlas
    en "desconocido" sería tirar el país, que sí sabemos.
    """
    out = df.copy()
    partido = out["vive_en"].map(_partir_vive_en)
    ciudad = partido.str[0]
    pais = partido.str[1]

    # PROBADO Y DESCARTADO: rescatar las 2.812 opiniones que traen "Ciudad -" con el
    # país vacío ("Madrid -", "Zaragoza -"), mirando qué ciudades aparecen con España
    # en el resto del dataset. Corrige ~350 filas españolas que hoy caen en "Otros",
    # pero rompe 3.938: alcanza con que exista una sola fila "Mexico - España" para
    # que las 3.673 opiniones que dicen "Mexico" pasen a ser España, y lo mismo con
    # Ecuador (265), "Berlín -" (66) y "London -" (1). El remedio es diez veces peor
    # que la enfermedad, así que las huérfanas quedan en "Otros".

    region = pd.Series(DESCONOCIDO, index=out.index, dtype=object)
    es_espania = pais.isin(("españa", "espana", "spain"))

    region[es_espania] = "España (sin especificar)"
    region[es_espania & ciudad.notna()] = "España castellanohablante"
    region[es_espania & ciudad.isin(CIUDADES_COOFICIALES)] = "España lengua cooficial"
    region[pais.eq("argentina")] = "Argentina"
    region[pais.isin(PAISES_SUDAMERICA)] = "Resto de Sudamérica"
    region[pais.notna() & (region == DESCONOCIDO)] = "Otros"

    out[COL_PAIS] = region
    return out


def imputar_genero_lector(df: pd.DataFrame) -> pd.DataFrame:
    """Completa el género del lector desde el nombre de pila.

    `genero_lector` trae el literal "-" en el 32% de las opiniones: es un nulo
    disfrazado que pandas no detecta, así que primero hay que reconocerlo como nulo.

    Los ambiguos, los nombres que no son nombres (`giovanniro255`, `j2c2`, `141008`) y
    los que no están en el diccionario van a "Desconocido". No es una derrota: en el
    análisis exploratorio ese grupo mostró una tasa de gusto del 81% contra 86% y 84%
    de hombres y mujeres, así que la ausencia del dato es informativa por sí misma.
    """
    out = df.copy()
    declarado = out["genero_lector"].where(~out["genero_lector"].isin(["-"]))

    pila = out["nombre"].str.strip().str.lower().str.split().str[0]
    es_nombre = pila.notna() & ~pila.str.contains(r"\d", na=True) & pila.str.len().ge(2)
    imputado = pila.where(es_nombre).map(GENERO_POR_NOMBRE).map({"M": "Hombre", "F": "Mujer"})

    out[COL_GENERO] = declarado.fillna(imputado).fillna(DESCONOCIDO)
    return out


def imputar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena las categóricas restantes con "Desconocido", no con la moda.

    Imputar con la moda inventa un dato: convierte "no sabemos la editorial" en
    "la editorial es Debolsillo", que es falso en la enorme mayoría de los casos y le
    da al modelo una señal que no existe. "Desconocido" es una categoría legítima y
    deja que el modelo decida si la ausencia significa algo.

    De paso normaliza mayúsculas y espacios: `genero_libro` tiene 62 categorías que
    son 53 ("No Ficción" y "No ficción", "HIstórica y aventuras"), y esa diferencia
    es tipeo, no información.
    """
    out = df.copy()
    for col in ["genero_libro", "editorial", "autor"]:
        normalizada = (out[col].str.strip().str.lower()
                       # Los acentos también separan categorías que son la misma:
                       # "clásicos" y "clasicos", "biografías" y el typo "biografiás".
                       # NFKD + descarte de los diacríticos lo resuelve con pandas y
                       # el codec ascii de la stdlib, sin `unidecode` (prohibida).
                       .str.normalize("NFKD")
                       .str.encode("ascii", "ignore").str.decode("ascii"))
        out[col] = normalizada.replace(list(NULOS_DISFRAZADOS), np.nan).fillna(DESCONOCIDO)
    return out


# --------------------------------------------------------------------------------------
# 4. Dummies
# --------------------------------------------------------------------------------------

# Sólo se hacen dummies de las categóricas de cardinalidad cerrada y chica. `autor`
# (19.551) y `editorial` (2.676) necesitan un top-N, y CLAUDE.md 3.3 exige que ese
# top-N se defina con value_counts() sobre train: va en la rama del split, no acá.
COLS_DUMMIES = [COL_PAIS, COL_GENERO, "genero_libro"]

# La guarda de idempotencia de `crear_dummies` compara por prefijo, así que si el
# nombre de una columna fuera prefijo de otra ("genero_lector" y
# "genero_lector_imputado"), la segunda nunca recibiría sus dummies. Hoy no pasa;
# esto lo deja explícito para que no pase en silencio si se agrega una columna.
assert not any(a != b and a.startswith(f"{b}_") for a in COLS_DUMMIES for b in COLS_DUMMIES), \
    "Un nombre de COLS_DUMMIES es prefijo de otro: la guarda por prefijo fallaría."


def crear_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las categóricas de baja cardinalidad en columnas numéricas.

    Sin este paso, todo el trabajo sobre región, género del lector y género literario
    es invisible para el modelo, que sólo mira columnas numéricas.
    """
    out = df.copy()
    # Idempotente a propósito: el runner de experimentos aplica dummies a cada etapa
    # y el pipeline las aplica de nuevo al final. Sin esta guarda, la segunda pasada
    # duplica cada columna en silencio y el modelo entrena con features repetidas.
    presentes = [c for c in COLS_DUMMIES
                 if c in out.columns
                 and not any(col.startswith(f"{c}_") for col in out.columns)]
    if not presentes:
        return out
    dummies = pd.get_dummies(out[presentes], prefix=presentes, dtype="int8")
    return pd.concat([out, dummies], axis=1)


# --------------------------------------------------------------------------------------
# El pipeline, legible de arriba abajo
# --------------------------------------------------------------------------------------
# El orden es el de CLAUDE.md 3.5. Los pasos que empeoraron la métrica quedan
# comentados con el motivo, no borrados: la función sigue disponible para volver a
# probarla cuando cambien las variables.

PIPELINE = [
    # Delta marginal 0.0000. Hoy no descarta ninguna fila porque no hay ratings nulos.
    # Queda como guarda: si en otra corrida aparecen, no tienen que entrar.
    filtrar_sin_target,

    # descartar_no_libros,
    #   Delta marginal -0.0023 (dentro del ruido, pero negativo). Descarta 666 filas
    #   cuyo libro no existe en el catálogo. Además contradice la decisión de unir con
    #   how="left": si se activa, el dataset pasa a ser el del inner. No queda.

    # Delta marginal -0.0030, dentro del ruido. Se mantiene igualmente: un lector de
    # 116 años no es un dato con el que el modelo deba decidir, y el 1910 es un
    # artefacto del formulario. Corrige el valor, no la métrica.
    anio_nacimiento_a_nulo,

    # Delta marginal -0.0043, dentro del ruido. Se mantiene porque es el paso que
    # convierte `anio_edicion` de texto a número: sin él la columna no existe para el
    # modelo, y las ramas siguientes la necesitan como materia prima (antigüedad del
    # libro, distancia entre edición y lectura).
    anio_edicion_a_nulo,

    # edicion_posterior_a_nulo,
    #   Delta marginal -0.0012. Anula el año de edición en el 7,4% de las opiniones a
    #   cambio de nada. El dato es conceptualmente incorrecto (el sitio guarda la
    #   última edición, no la leída), pero el modelo no mejora al sacarlo. No queda.

    # Delta marginal +0.0052: el único paso de limpieza que supera la banda de ruido.
    normalizar_vive_en,

    # Delta marginal +0.0042, apenas por debajo del umbral. Se mantiene: recupera el
    # género de 89.103 opiniones y deja "Desconocido" como categoría, que el análisis
    # exploratorio mostró que discrimina (81% de gusto contra 86% y 84%).
    imputar_genero_lector,

    # Delta marginal +0.0007. Se mantiene por parsimonia, no por métrica: colapsa las
    # categorías que sólo diferían en mayúsculas y acentos, y el modelo termina con 11
    # columnas menos para el mismo f1.
    imputar_categoricas,

    # Delta marginal +0.0155: el cambio más grande de toda la rama. Sin dummies, todo
    # el trabajo sobre categóricas es invisible para el modelo.
    crear_dummies,
]


def aplicar(df: pd.DataFrame, funciones=None) -> pd.DataFrame:
    """Aplica las funciones en orden. Sin argumento, corre el pipeline completo."""
    out = df
    for funcion in (PIPELINE if funciones is None else funciones):
        out = funcion(out)
    return out
