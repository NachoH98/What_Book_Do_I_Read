# TP Individual — QuéLibroLeo · E72.1.01 Fundamentos de Métodos Analíticos Predictivos

Contexto permanente del proyecto. Leer siempre antes de escribir código.

---

## 1. El problema

Predecir si a un lector le va a gustar un libro que no leyó, a partir de los datos del
sitio **quelibroleo.com** provistos por la cátedra.

Es un problema de **clasificación binaria**, no de regresión.

| Rating | Clase |
|---|---|
| ≥ 7 | `1` — le gustó |
| 6 | **se descarta** (rating gris, no aporta información) |
| ≤ 5 | `0` — no le gustó |

El orden importa: **primero se eliminan los registros con rating 6, después se mapea.**

Distribución esperada: **≈ 80/20** a favor de "le gustó". Es desbalanceo clásico,
no extremo.

---

## 2. Los datos: tres tablas

Los datos **no están en el repositorio** (están en Drive, y `data/` está en `.gitignore`).
El código los lee desde una ruta configurable.

**libros** — `id_libro`, `titulo`, `autor`, `genero`, `editorial`, `anio_edicion`, `isbn`, `resumen`, `img_src`
**lectores** — `id_lector`, `nombre`, `genero`, `vive_en`, `nacimiento`
**opiniones** — `id_lector`, `id_libro`, `fecha` (dd/mm/aaaa), `rating`

### Regla de unión

La tabla de **opiniones es el centro**. Cada fila del dataset final es una opinión.

```python
base = (opiniones
        .merge(libros,   on="id_libro",  how="left")
        .merge(lectores, on="id_lector", how="left"))
```

⚠ **Sólo se limpia lo que aparece en las opiniones.** Un libro sin ninguna opinión no
entra al dataset, así que sus nulos no son un problema a resolver: desaparecen solos con
el merge. Lo mismo con los lectores. Aproximadamente el 60% de la tabla de libros está
vacía y en su mayoría corresponde a libros sin interacciones.

Al hacer el merge hay que **decidir explícitamente** qué pasa con una opinión cuyo libro
o lector no existe en las otras tablas, y documentar la decisión (`how` = `left` vs `inner`).

---

## 3. ⚠ Restricciones de la cátedra — no negociables

### 3.1 Sólo librerías vistas en clase

**Permitidas:**
`pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy` (`stats`, `cluster.hierarchy`) ·
`scikit-learn` · `imbalanced-learn` (`imblearn`) · `mlxtend` · `spacy` (`es_core_news_sm`) ·
`nltk` · `wordcloud`

**Prohibidas, sin excepción:** `xgboost`, `lightgbm`, `catboost`, `optuna`, `hyperopt`,
`shap`, `lime`, `pycaret`, `gender-guesser`, `unidecode`, y cualquier otra que no se haya
visto en clase. Usarlas obliga a recuperar la materia.

**Patrón para necesidades cubiertas por librerías prohibidas:** ejecutar la herramienta
**una sola vez, fuera del entregable**, y pegar el resultado como un diccionario dentro
del código.

```python
# Generado offline; NO se importa ninguna librería externa
GENERO_POR_NOMBRE = {"jorge": "M", "carla": "F", "alex": "D", ...}
df["genero"] = df["genero"].fillna(df["nombre_pila"].str.lower().map(GENERO_POR_NOMBRE))
```

Documentar en el informe de dónde salió el mapeo.

### 3.2 Una sola métrica de decisión

El dataset está desbalanceado, así que **`accuracy` está prohibida**.

Elegir **una** entre `precision`, `recall` y `f1`, declararla al principio del informe y
justificarla en dos líneas. **No se acepta reportar ocho métricas y elegir la que
convenga en cada comparación.** Se puede mostrar la matriz de confusión como material
descriptivo, pero las decisiones se toman siempre con la misma métrica.

Recomendación por defecto: **`f1`**, salvo que haya un argumento de negocio para preferir
precisión o recall.

### 3.3 Nada de fuga de información

- **El split va antes que cualquier estadística.** Los perfiles de usuario y de libro
  (% de libros que le gustaron por género, etc.) se calculan **sólo sobre `train`** y se
  aplican a ambos conjuntos.
- El **top-N** de autores y editoriales se define con `value_counts()` sobre `train`.
- El **remuestreo** por desbalanceo va **sólo sobre train**, dentro de
  `imblearn.pipeline.Pipeline` (no `sklearn.pipeline.Pipeline`).
- La columna `rating` y cualquier derivado suyo **no pueden ser variables predictoras**:
  contienen el target.

### 3.4 Ceteris paribus: un cambio por vez

Es la regla metodológica central del curso.

Cada cambio (una limpieza, una variable nueva) se prueba **solo**, contra el mismo modelo
congelado, y se registra en la tabla de resultados. Si mejora, queda. Si no, **se comenta
en el código, no se borra.**

Cambiar cinco cosas y correr el modelo no permite saber cuál funcionó, y es el error que
la cátedra marca como "de novato".

### 3.5 Orden de las operaciones

1. **Filtrar** (qué registros entran al dataset)
2. **Outliers → nulo** (detectar y poner en `NaN`, no imputar todavía)
3. **Imputar** (con lógica del propio registro cuando se pueda, no con la media por defecto)
4. **Dummies** (las categóricas a numéricas)
5. **Crear variables** (la parte creativa, de a una)
6. **Split**, y recién después estadísticas de perfil, remuestreo y modelos

**Limpiar todo primero, balancear después.**

---

## 4. Jerarquía de impacto

| Etapa | Cuánto baja el error |
|---|---|
| Limpieza | Un poco |
| **Creación de variables** | **Muchísimo** |
| Optimización de hiperparámetros | No tanto (está automatizada) |

El esfuerzo se concentra en la creación de variables.

---

## 5. Reglas de trabajo prácticas

- **Checkpoints en disco, no copias en RAM.** `df.to_pickle("checkpoints/xx.pkl")` y
  `pd.read_pickle(...)`. Pickle conserva los tipos, es más rápido y ocupa menos que CSV.
  Nunca `df2 = df.copy()` sobre datasets grandes.
- **Muestra para iterar.** Trabajar con el 25% para probar ideas; lo que mejora en la
  muestra normalmente mejora en el total. La versión final corre con el 100%.
- **`%%time`** al principio de las celdas de modelos: la cátedra pide el tiempo de
  ejecución en la tabla de resultados.
- **Normalizar entre 0 y 1** si se usa KNN o cualquier modelo basado en distancias.
- **`random_state=42`** en todo: split, modelos, samplers, muestreos. Sin eso los errores
  no son comparables entre corridas.
- **Reglas generales, nunca fila por fila.** Nada de `df.loc[3438, "col"] = x`.
- **Esfuerzo contra beneficio**: no invertir dos horas en limpiar 50 registros de decenas
  de miles.

---

## 6. Estructura del repositorio

```
.
├── CLAUDE.md
├── PROMPTS.md
├── .gitignore              # data/, checkpoints/, *.pkl, .venv/
├── data/                   # NO versionado — los tres CSV van acá
├── checkpoints/            # NO versionado
├── src/
│   ├── config.py           # rutas, semilla, métrica, constantes
│   ├── carga.py            # lectura y unión de las tres tablas
│   ├── limpieza.py         # funciones de limpieza, una por transformación
│   ├── variables.py        # creación de atributos
│   ├── texto.py            # minería de texto sobre `resumen`
│   ├── modelo.py           # modelo congelado de evaluación + pipeline final
│   └── experimentos.py     # el runner que produce la tabla de resultados
├── resultados/
│   ├── tabla_experimentos.csv
│   └── figuras/
└── notebooks/
    └── TP_individual.ipynb # se arma al final, para subir a Colab
```

**Se desarrolla en módulos `.py`** (los diffs de git son legibles y las funciones se
reutilizan). El notebook único se ensambla en la última rama.

---

## 7. Estilo de código

- Funciones con una sola responsabilidad, que **reciben un DataFrame y devuelven uno
  nuevo**, sin mutar el original ni depender de estado global.
- Cada función de limpieza o de creación de variables debe poder activarse o desactivarse
  de forma independiente: así el runner de experimentos puede probarlas de a una.
- Docstring corto en español explicando **qué decide** la función, no qué hace línea por
  línea.
- Nada de código muerto sin comentar. Lo que se probó y no funcionó queda comentado con
  una nota de por qué.

---

## 8. La entrega

**Dos enlaces en la plataforma de la ENAP:**

1. **Un** Google Colab (uno solo, no tres), ejecutado, con secciones y títulos,
   comentado, sin celdas basura tipo `df.shape` sueltas.
2. **Un** Google Doc con el informe.

Ambos compartidos como **"Lector · cualquiera con el vínculo"**, desde el propio Colab,
no desde Drive. **Verificar en ventana de incógnito antes de entregar.** Hacer copia de
ambos documentos antes de entregar.

### El informe

- **Estilo paper: se cuenta la victoria, no todo lo que se hizo.** Las pruebas fallidas
  van a un anexo.
- **8 a 10 páginas máximo.**
- Secciones mínimas: **Introducción, Resultados, Conclusiones**.
- **Tablas y gráficos, no párrafos.** Escribir resultados en prosa baja puntos. Los folds
  de la validación cruzada van en un **boxplot**, no escritos.
- El informe **no lleva código**; el código **no lleva texto de análisis**.
- Sin errores de ortografía ni gramática.

### La documentación del Colab

Comentar **la jugada**, no traducir el código al castellano. "Acá optimizo los
hiperparámetros; los mejores son estos" — no "a la variable `a` le sumo uno".
