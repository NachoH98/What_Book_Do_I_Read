"""Análisis exploratorio. No modifica los datos: sólo mira y produce figuras.

Cada figura tiene que poder acompañarse de un análisis; si no dice nada, no está.
Por eso no hay un gráfico por columna: hay un gráfico por hallazgo.

Uso:  python -m src.eda
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import carga, config

# --------------------------------------------------------------------------------------
# Estilo
# --------------------------------------------------------------------------------------
# Paleta validada para daltonismo (ΔE CVD 9.2, visión normal 24.0 sobre las tres
# series). El orden de los slots es el mecanismo de seguridad, no decoración.

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
GRILLA = "#e3e2de"

AZUL = "#2a78d6"      # serie 1 — el color por defecto de todo lo univariado
NARANJA = "#eb6834"   # serie 2
AQUA = "#1baf7a"      # serie 3
ROJO = "#e34948"      # reservado: marca un problema de calidad, siempre con etiqueta
GRIS = "#9a9892"      # contexto, referencia, "resto"

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": GRILLA,
    "axes.labelcolor": TINTA_SUAVE,
    "axes.titlecolor": TINTA,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRILLA,
    "grid.linewidth": 0.8,
    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "text.color": TINTA,
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _limpiar(ax, ejes=("top", "right")) -> None:
    """Saca el marco: la grilla ya da la referencia, el marco sólo agrega tinta."""
    for lado in ejes:
        ax.spines[lado].set_visible(False)
    ax.grid(axis="x", visible=False)


# Contorno del color de la superficie: despega una etiqueta de la barra o de la
# línea de referencia sobre la que caiga, sin recuadro ni fondo opaco.
HALO = [pe.withStroke(linewidth=2.6, foreground=SUPERFICIE)]


def _titulo(fig, texto: str) -> None:
    """Título de la figura, reservando el espacio antes de escribirlo.

    Sin el `rect` el suptitle se superpone con el título del primer eje.
    """
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle(texto, x=0.005, ha="left", fontsize=13, fontweight="bold", y=0.995)


def _guardar(fig, nombre: str) -> None:
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_FIGURAS / f"{nombre}.png"
    fig.savefig(destino)
    # `show` antes de `close`: en el notebook dibuja la figura debajo de la celda, y
    # con el backend Agg del script es un no-op. Sin esto, el Colab no muestra ni un
    # gráfico: las figuras se irían al disco y el lector no vería nada.
    plt.show()
    plt.close(fig)
    print(f"  → {destino.name}")


def _etiquetar_barras(ax, barras, valores, fmt="{:.0%}", dx=0.0) -> None:
    """Etiqueta directa sobre cada barra: sin hover, el número tiene que estar."""
    for barra, valor in zip(barras, valores):
        ax.text(barra.get_width() + dx, barra.get_y() + barra.get_height() / 2,
                fmt.format(valor), va="center", ha="left",
                fontsize=8, color=TINTA_SUAVE, path_effects=HALO)


# --------------------------------------------------------------------------------------
# 1. Univariado: el target
# --------------------------------------------------------------------------------------

def fig_rating_y_target(opiniones: pd.DataFrame, base: pd.DataFrame) -> None:
    """El hueco del rating 6 y el desbalanceo que deja.

    Justifica las dos reglas del target: por qué se descarta el 6 y por qué la
    métrica no puede ser accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), width_ratios=[2, 1])

    conteo = opiniones[config.COL_RATING].value_counts().sort_index()
    colores = [ROJO if r == config.RATING_DESCARTADO else AZUL for r in conteo.index]
    ax1.bar(conteo.index, conteo.values, color=colores, width=0.7)
    ax1.set_title("Distribución del rating, antes de construir el target")
    ax1.set_xlabel("rating"); ax1.set_ylabel("opiniones")
    ax1.set_xticks(range(1, 11))
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    descartadas = int(conteo.get(config.RATING_DESCARTADO, 0))
    ax1.annotate(f"rating 6 — se descarta\n{descartadas:,} opiniones ({descartadas/len(opiniones):.1%})",
                 xy=(6, descartadas), xytext=(3.2, descartadas * 1.06),
                 color=ROJO, fontsize=8.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=ROJO, lw=1.2))
    _limpiar(ax1)

    dist = base[config.TARGET].value_counts().sort_index()
    barras = ax2.bar(["0 · no gustó", "1 · gustó"], dist.values, color=[NARANJA, AZUL], width=0.6)
    ax2.set_title("Target `gusto`")
    ax2.set_ylabel("opiniones")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    for barra, valor in zip(barras, dist.values):
        ax2.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor/len(base):.1%}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=TINTA)
    ax2.set_ylim(0, dist.max() * 1.15)
    _limpiar(ax2)

    _titulo(fig, "El rating gris se elimina y deja un problema desbalanceado 84/16")
    _guardar(fig, "01_rating_y_target")


# --------------------------------------------------------------------------------------
# 2. Univariado: las numéricas, y sus valores imposibles
# --------------------------------------------------------------------------------------

def fig_nacimiento_y_edad(base: pd.DataFrame) -> None:
    """El pico de 1910 es un centinela del formulario, no gente de 116 años."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    lectores = base.drop_duplicates("id_lector")
    conteo = lectores.nacimiento.value_counts().sort_index()
    colores = [ROJO if a == 1910 else AZUL for a in conteo.index]
    ax1.bar(conteo.index, conteo.values, color=colores, width=1.0)
    ax1.set_title("Año de nacimiento declarado (lectores únicos)")
    ax1.set_xlabel("año de nacimiento"); ax1.set_ylabel("lectores")
    pico = int(conteo.get(1910, 0))
    ax1.annotate(f"1910: {pico} lectores\nes el mínimo del selector del sitio,\nno una cohorte real",
                 xy=(1910, pico), xytext=(1925, pico * 0.85),
                 color=ROJO, fontsize=8.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=ROJO, lw=1.2))
    _limpiar(ax1)

    edad = pd.to_datetime(base.fecha, format="%d-%m-%Y").dt.year - base.nacimiento
    ax2.hist(edad.dropna(), bins=np.arange(-10, 125, 2), color=AZUL)
    ax2.set_title("Edad del lector al momento de opinar")
    ax2.set_xlabel("edad (años)"); ax2.set_ylabel("opiniones")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    for limite, etiqueta in [(10, "menores de 10"), (90, "90 o más")]:
        ax2.axvline(limite, color=ROJO, lw=1.2, ls="--")
    n_bajo = int((edad < 10).sum()); n_alto = int((edad >= 90).sum())
    ax2.annotate(f"{n_bajo:,} opiniones con edad < 10\n"
                 f"{n_alto:,} con edad ≥ 90 — son la joroba aislada\n"
                 f"de la derecha, casi toda del centinela 1910",
                 xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
                 color=ROJO, fontsize=8.5, fontweight="bold", path_effects=HALO)
    _limpiar(ax2)

    _titulo(fig, "`nacimiento` tiene un valor centinela que contamina toda la edad")
    _guardar(fig, "02_nacimiento_y_edad")


def fig_anio_edicion(base: pd.DataFrame) -> None:
    """La edición que guarda el sitio es la última, no la que leyó el lector."""
    anio = pd.to_numeric(base.anio_edicion, errors="coerce")
    anio_opinion = pd.to_datetime(base.fecha, format="%d-%m-%Y").dt.year

    fig, ax = plt.subplots(figsize=(9, 3.8))
    validos = anio[(anio >= 1900) & (anio <= 2026)]
    ax.hist(validos, bins=np.arange(1900, 2028, 1), color=AZUL)
    ax.set_title("Año de edición del libro")
    ax.set_xlabel("año de edición"); ax.set_ylabel("opiniones")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")

    posteriores = int((anio > anio_opinion).sum())
    fuera = int(((anio < 1900) | (anio > 2026)).sum() + (anio.isna() & base.anio_edicion.notna()).sum())
    ax.annotate(
        f"{posteriores:,} opiniones ({posteriores/len(base):.1%}) tienen un año de edición\n"
        f"POSTERIOR a la fecha de la opinión: el sitio guarda la última edición\n"
        f"del catálogo, no el ejemplar que se leyó.\n"
        f"Otras {fuera:,} tienen el año ilegible o fuera de rango.",
        xy=(0.02, 0.95), xycoords="axes fraction", va="top",
        color=ROJO, fontsize=8.5, fontweight="bold")
    _limpiar(ax)
    _titulo(fig, "`anio_edicion` no es el año en que se leyó el libro")
    _guardar(fig, "03_anio_edicion")


# --------------------------------------------------------------------------------------
# 3. Univariado: frecuencias categóricas
# --------------------------------------------------------------------------------------

def fig_frecuencias_categoricas(base: pd.DataFrame) -> None:
    """Qué domina cada categórica. Muestra la concentración y la cola sucia."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    paneles = [
        (axes[0][0], "genero_libro", "Género literario (top 12)", 12),
        (axes[0][1], "genero_lector", "Género del lector — tal como viene", None),
        (axes[1][0], "editorial", "Editorial (top 12)", 12),
        (axes[1][1], "autor", "Autor (top 12)", 12),
    ]
    for ax, col, titulo, top in paneles:
        conteo = base[col].value_counts(dropna=False)
        if top:
            conteo = conteo.head(top)
        etiquetas = [("(nulo)" if pd.isna(i) else str(i))[:38] for i in conteo.index]
        colores = [ROJO if (pd.isna(i) or str(i).strip() in {"-", "¿?"}) else AZUL
                   for i in conteo.index]
        barras = ax.barh(etiquetas[::-1], conteo.values[::-1], color=colores[::-1], height=0.72)
        ax.set_title(titulo)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
        _etiquetar_barras(ax, barras, conteo.values[::-1] / len(base),
                          dx=conteo.max() * 0.01)
        ax.set_xlim(0, conteo.max() * 1.18)
        ax.tick_params(labelsize=8)
        _limpiar(ax)

    axes[0][1].annotate('el literal "-" es un nulo disfrazado',
                        xy=(0.35, 0.15), xycoords="axes fraction",
                        color=ROJO, fontsize=8.5, fontweight="bold")
    _titulo(fig, "Frecuencias por categoría — en rojo, los nulos disfrazados")
    _guardar(fig, "04_frecuencias_categoricas")


# --------------------------------------------------------------------------------------
# 4. Cardinalidad: lo que condiciona todo lo que venga después
# --------------------------------------------------------------------------------------

def fig_cardinalidad(base: pd.DataFrame) -> pd.DataFrame:
    """Cuánta cobertura da el top-N de cada categórica.

    Es la figura que decide la estrategia de dummies: con 19.500 autores no se
    puede hacer one-hot, pero si el top 100 cubre buena parte de las opiniones,
    un top-N + "otros" sí es viable.
    """
    columnas = [("autor", AZUL), ("editorial", NARANJA), ("genero_libro", AQUA)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    filas = []
    for col, color in columnas:
        conteo = base[col].value_counts()
        acumulado = conteo.cumsum() / len(base)
        x = np.arange(1, len(acumulado) + 1)
        ax.plot(x, acumulado.values, color=color, lw=2, solid_capstyle="round")
        # Etiqueta directa al final de la curva: el aqua no llega a 3:1 de
        # contraste, así que la identidad no puede depender sólo del color.
        ax.annotate(f"{col} ({conteo.size:,})",
                    xy=(len(acumulado), acumulado.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", va="center")
        fila = {"columna": col, "categorías": conteo.size}
        for n in (10, 50, 100):
            fila[f"top {n}"] = round(float(acumulado.iloc[min(n, len(acumulado)) - 1]), 4)
        filas.append(fila)

    for n in (10, 50, 100):
        ax.axvline(n, color=GRIS, lw=0.9, ls=":")
        ax.text(n, 1.02, f"top {n}", ha="center", fontsize=8, color=TINTA_SUAVE)

    ax.set_xscale("log")
    ax.set_xlim(1, 3e4)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("cantidad de categorías incluidas (escala logarítmica)")
    ax.set_ylabel("cobertura de las opiniones")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Cobertura acumulada del top-N de cada categórica")
    _limpiar(ax)
    _titulo(fig, "La cardinalidad decide la estrategia de dummies")
    _guardar(fig, "05_cardinalidad_acumulada")
    return pd.DataFrame(filas).set_index("columna")


# --------------------------------------------------------------------------------------
# 5. Actividad: la cola larga
# --------------------------------------------------------------------------------------

def fig_actividad(base: pd.DataFrame) -> pd.DataFrame:
    """Opiniones por lector y por libro. La cola es el hallazgo, no el promedio."""
    por_lector = base.id_lector.value_counts()
    por_libro = base.id_libro.value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, serie, etiqueta, color in [(ax1, por_lector, "lector", AZUL),
                                       (ax2, por_libro, "libro", NARANJA)]:
        bins = np.logspace(0, np.log10(serie.max()) + 0.05, 45)
        ax.hist(serie.values, bins=bins, color=color)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"Opiniones por {etiqueta}")
        ax.set_xlabel(f"opiniones del {etiqueta} (log)")
        ax.set_ylabel(f"cantidad de {etiqueta}es (log)" if etiqueta == "lector"
                      else "cantidad de libros (log)")
        ax.annotate(
            f"mediana {serie.median():.0f} · máximo {serie.max():,}\n"
            f"{(serie == 1).mean():.0%} tiene una sola opinión",
            xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=8.5, color=TINTA_SUAVE, path_effects=HALO)
        _limpiar(ax)
        ax.grid(axis="x", visible=True)

    _titulo(fig, "La actividad es una cola larga: pocos lectores explican muchas opiniones")
    _guardar(fig, "06_actividad_lector_y_libro")

    resumen = pd.DataFrame({
        "opiniones por lector": por_lector.describe(percentiles=[.5, .9, .99]),
        "opiniones por libro": por_libro.describe(percentiles=[.5, .9, .99]),
    }).round(1)
    return resumen


# --------------------------------------------------------------------------------------
# 6. Bivariado contra el target
# --------------------------------------------------------------------------------------

def _tasa_por(base: pd.DataFrame, col: pd.Series, minimo: int = 500) -> pd.DataFrame:
    """Tasa de `gusto` por categoría, descartando las categorías sin volumen.

    Una categoría con 12 opiniones puede dar 100% de gusto por azar; incluirla
    ensucia el gráfico y sugiere una relación que no existe.
    """
    tabla = (base.assign(_cat=col)
             .groupby("_cat", dropna=False)[config.TARGET]
             .agg(tasa="mean", n="size"))
    return tabla[tabla.n >= minimo].sort_values("tasa")


def fig_tasa_por_genero_literario(base: pd.DataFrame) -> pd.DataFrame:
    """El género del libro sí mueve la aguja: hay 20 puntos entre extremos."""
    tabla = _tasa_por(base, base.genero_libro, minimo=1000)
    global_ = base[config.TARGET].mean()

    fig, ax = plt.subplots(figsize=(9, 6))
    colores = [AZUL if t >= global_ else NARANJA for t in tabla.tasa]
    barras = ax.barh([str(i)[:40] for i in tabla.index], tabla.tasa, color=colores, height=0.72)
    ax.axvline(global_, color=TINTA_SUAVE, lw=1.4, ls="--")
    ax.text(global_, len(tabla) - 0.2, f"  media global {global_:.1%}",
            fontsize=8.5, color=TINTA_SUAVE, va="top", path_effects=HALO)
    _etiquetar_barras(ax, barras, tabla.tasa.values, dx=0.004)
    for barra, n in zip(barras, tabla.n.values):
        ax.text(0.006, barra.get_y() + barra.get_height() / 2, f"n={n:,}",
                va="center", fontsize=7.5, color=SUPERFICIE)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("tasa de «le gustó»")
    ax.set_title("Tasa de «le gustó» por género literario (categorías con n ≥ 1.000)")
    ax.tick_params(labelsize=8.5)
    _limpiar(ax)
    rango = (tabla.tasa.max() - tabla.tasa.min()) * 100
    _titulo(fig, f"El género del libro discrimina: {rango:.0f} puntos entre el mejor "
                 f"({tabla.index[-1]}) y el peor ({tabla.index[0]})")
    _guardar(fig, "07_tasa_gusto_por_genero_literario")
    return tabla


def fig_tasa_por_decada(base: pd.DataFrame) -> pd.DataFrame:
    """La década de edición contra el target."""
    anio = pd.to_numeric(base.anio_edicion, errors="coerce")
    anio = anio.where((anio >= 1900) & (anio <= 2026))
    decada = (anio // 10 * 10)
    tabla = _tasa_por(base, decada, minimo=500)
    tabla = tabla.sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(tabla.index, tabla.tasa, color=AZUL, lw=2, marker="o", ms=6,
            mfc=AZUL, mec=SUPERFICIE, mew=1.5)
    ax.axhline(base[config.TARGET].mean(), color=TINTA_SUAVE, lw=1.2, ls="--")
    ax.text(tabla.index.max(), base[config.TARGET].mean(),
            f"media global {base[config.TARGET].mean():.1%} ", fontsize=8.5,
            color=TINTA_SUAVE, va="top", ha="right", path_effects=HALO)
    for x, y, n in zip(tabla.index, tabla.tasa, tabla.n):
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=TINTA_SUAVE, path_effects=HALO)
    # La escala NO se ajusta al rango de los datos: con 1,9 puntos de diferencia, un
    # eje pegado a los datos dibujaría una montaña y sugeriría un efecto que no está.
    # El eje fijo muestra lo que hay, que es una recta.
    ax.set_ylim(0.6, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("década de edición"); ax.set_ylabel("tasa de «le gustó»")
    ax.set_title("Tasa de «le gustó» por década de edición (n ≥ 500)")
    _limpiar(ax)
    ax.grid(axis="x", visible=True)
    rango = (tabla.tasa.max() - tabla.tasa.min()) * 100
    _titulo(fig, f"La década de edición casi no mueve la aguja: {rango:.1f} puntos "
                 f"entre extremos")
    _guardar(fig, "08_tasa_gusto_por_decada")
    return tabla


def fig_tasa_por_perfil_lector(base: pd.DataFrame) -> pd.DataFrame:
    """Género del lector y actividad del lector contra el target."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), width_ratios=[1, 1.4])
    global_ = base[config.TARGET].mean()

    genero = _tasa_por(base, base.genero_lector, minimo=500)
    colores = [ROJO if str(i).strip() == "-" else AZUL for i in genero.index]
    barras = ax1.barh([("(nulo)" if pd.isna(i) else str(i)) for i in genero.index],
                      genero.tasa, color=colores, height=0.6)
    ax1.axvline(global_, color=TINTA_SUAVE, lw=1.4, ls="--")
    _etiquetar_barras(ax1, barras, genero.tasa.values, dx=0.004)
    ax1.set_xlim(0, 1.0)
    ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax1.set_title("Por género del lector")
    ax1.set_xlabel("tasa de «le gustó»")
    _limpiar(ax1)

    por_lector = base.id_lector.value_counts()
    cortes = [0, 1, 5, 20, 100, 500, np.inf]
    etiquetas = ["1", "2–5", "6–20", "21–100", "101–500", "500+"]
    tramo = pd.cut(base.id_lector.map(por_lector), bins=cortes, labels=etiquetas)
    actividad = _tasa_por(base, tramo, minimo=500).reindex(etiquetas).dropna()

    ax2.plot(range(len(actividad)), actividad.tasa, color=NARANJA, lw=2,
             marker="o", ms=7, mfc=NARANJA, mec=SUPERFICIE, mew=1.5)
    ax2.axhline(global_, color=TINTA_SUAVE, lw=1.2, ls="--")
    ax2.set_xticks(range(len(actividad)))
    ax2.set_xticklabels(actividad.index)
    for i, (y, n) in enumerate(zip(actividad.tasa, actividad.n)):
        ax2.annotate(f"{y:.0%}\nn={n/1000:.0f}k", (i, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color=TINTA_SUAVE,
                     path_effects=HALO)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.set_xlabel("opiniones totales del lector")
    ax2.set_title("Por actividad del lector")
    ax2.set_ylim(actividad.tasa.min() - 0.06, actividad.tasa.max() + 0.09)
    _limpiar(ax2)
    ax2.grid(axis="x", visible=True)

    _titulo(fig, "El lector que más opina es más exigente: la tasa cae con la actividad")
    _guardar(fig, "09_tasa_gusto_por_perfil_lector")
    return actividad


# --------------------------------------------------------------------------------------
# 7. Calidad de datos, con evidencia
# --------------------------------------------------------------------------------------

def fig_calidad(base: pd.DataFrame) -> pd.DataFrame:
    """Inventario de problemas detectados, medido en opiniones afectadas."""
    anio = pd.to_numeric(base.anio_edicion, errors="coerce")
    anio_opinion = pd.to_datetime(base.fecha, format="%d-%m-%Y").dt.year
    edad = anio_opinion - base.nacimiento
    nombres_raros = base.nombre.fillna("").str.contains(r"\d", regex=True)

    problemas = [
        ('`genero_lector` con el literal "-"', int((base.genero_lector == "-").sum())),
        ("`nacimiento` nulo", int(base.nacimiento.isna().sum())),
        ("`nacimiento` = 1910 (centinela)", int((base.nacimiento == 1910).sum())),
        ("edad al opinar fuera de [10, 90]", int(((edad < 10) | (edad > 90)).sum())),
        ("`anio_edicion` posterior a la opinión", int((anio > anio_opinion).sum())),
        ("`vive_en` nulo", int(base.vive_en.isna().sum())),
        ('`vive_en` con el literal "¿?"', int((base.vive_en == "¿?").sum())),
        ("`vive_en` sin separador ciudad-país", int((base.vive_en.notna() & ~base.vive_en.str.contains(" - ", na=False)).sum())),
        ("`nombre` con dígitos (no es un nombre)", int(nombres_raros.sum())),
        ("libro inexistente (título nulo)", int(base.titulo.isna().sum())),
        ("`resumen` nulo", int(base.resumen.isna().sum())),
    ]
    tabla = (pd.DataFrame(problemas, columns=["problema", "opiniones"])
             .assign(**{"% del dataset": lambda t: (t.opiniones / len(base) * 100).round(2)})
             .sort_values("opiniones"))

    fig, ax = plt.subplots(figsize=(10, 5))
    barras = ax.barh(tabla.problema, tabla.opiniones, color=ROJO, height=0.7)
    _etiquetar_barras(ax, barras, tabla.opiniones / len(base), dx=len(base) * 0.004)
    ax.set_xlim(0, tabla.opiniones.max() * 1.2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")
    ax.set_xlabel("opiniones afectadas")
    ax.set_title("Problemas de calidad detectados, medidos en opiniones afectadas")
    ax.tick_params(labelsize=8.5)
    _limpiar(ax)
    _titulo(fig, "Qué hay para limpiar, y cuánto pesa cada cosa")
    _guardar(fig, "10_calidad_de_datos")
    return tabla.sort_values("opiniones", ascending=False).reset_index(drop=True)


def evidencia_calidad(base: pd.DataFrame) -> None:
    """Los ejemplos concretos que respaldan el gráfico anterior."""
    lectores = base.drop_duplicates("id_lector")

    print("\n— Nombres que no son nombres (rompen la imputación de género por nombre):")
    raros = lectores.nombre.dropna()
    raros = raros[raros.str.contains(r"\d", regex=True)]
    print(f"  {len(raros)} de {lectores.nombre.notna().sum()} lectores. Ejemplos: {raros.head(12).tolist()}")

    print("\n— `vive_en`: formatos mezclados en la misma columna:")
    v = base.vive_en.dropna()
    print(f"  con 'ciudad - país': {v.str.contains(' - ').mean():.1%}")
    ejemplos = ["Madrid - España", "España", "¿?", "Tenerife -", "France, French Republic"]
    for e in ejemplos:
        print(f"    {e!r:35} → {int((base.vive_en == e).sum()):,} opiniones")

    print("\n— Registros que no son libros:")
    libros = base.drop_duplicates("id_libro")
    sin_titulo = libros.titulo.isna().sum()
    print(f"  {sin_titulo} id_libro con opiniones no existen en la tabla de libros")
    print("  (no se encontró un volumen relevante de revistas/agendas mal catalogadas:")
    print("   el problema no es qué tipo de registro es, es que el registro no está)")

    print("\n— Género literario: la misma categoría escrita de varias formas:")
    g = base.genero_libro.dropna()
    print(f"  {g.nunique()} categorías → {g.str.lower().str.strip().nunique()} tras normalizar mayúsculas")
    conteo = g.value_counts()
    for cat in ["No Ficción", "No ficción", "Lecturas complementarias", "Lecturas Complementarias"]:
        if cat in conteo:
            print(f"    {cat!r:32} → {conteo[cat]:,}")


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def main() -> None:
    pd.set_option("display.width", 200)
    print("Cargando el dataset base y la tabla de opiniones cruda…")
    base = pd.read_pickle(config.CHECKPOINT_BASE)
    # La tabla cruda hace falta sólo para mostrar el hueco del rating 6, que en el
    # checkpoint ya no está.
    _, _, opiniones = carga.cargar_tablas()
    print(f"base: {len(base):,} filas · opiniones crudas: {len(opiniones):,}\n")

    print("Figuras:")
    fig_rating_y_target(opiniones, base)
    fig_nacimiento_y_edad(base)
    fig_anio_edicion(base)
    fig_frecuencias_categoricas(base)
    cardinalidad = fig_cardinalidad(base)
    actividad = fig_actividad(base)
    genero = fig_tasa_por_genero_literario(base)
    decada = fig_tasa_por_decada(base)
    perfil = fig_tasa_por_perfil_lector(base)
    calidad = fig_calidad(base)

    print("\n" + "=" * 88)
    print("CARDINALIDAD — cobertura de las opiniones por el top-N")
    print("=" * 88)
    print(cardinalidad.to_string())

    print("\n" + "=" * 88)
    print("ACTIVIDAD")
    print("=" * 88)
    print(actividad.to_string())

    print("\n" + "=" * 88)
    print("TASA DE «LE GUSTÓ» POR GÉNERO LITERARIO")
    print("=" * 88)
    print(genero.assign(tasa=lambda t: (t.tasa * 100).round(1)).to_string())

    print("\n" + "=" * 88)
    print("TASA POR DÉCADA Y POR ACTIVIDAD DEL LECTOR")
    print("=" * 88)
    print(decada.assign(tasa=lambda t: (t.tasa * 100).round(1)).to_string())
    print()
    print(perfil.assign(tasa=lambda t: (t.tasa * 100).round(1)).to_string())

    print("\n" + "=" * 88)
    print("CALIDAD DE DATOS")
    print("=" * 88)
    print(calidad.to_string(index=False))
    evidencia_calidad(base)

    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    calidad.to_csv(config.DIR_RESULTADOS / "03_problemas_de_calidad.csv", index=False)
    cardinalidad.to_csv(config.DIR_RESULTADOS / "03_cardinalidad.csv")
    print(f"\nTablas guardadas en {config.DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
