"""Mapa nombre de pila → género, generado una sola vez fuera del entregable.

CLAUDE.md 3.1 prohíbe `gender-guesser` y cualquier librería que no se haya visto en
clase, y propone el patrón de ejecutar la herramienta una vez afuera y pegar el
resultado como diccionario. Acá la "herramienta" no es una librería externa: es el
propio dataset. El mapa sale de los 7,056 lectores que SÍ declararon su género,
agrupando por nombre de pila.

Criterio de inclusión: el nombre aparece en al menos 2 lectores y al menos el 90% de
ellos declara el mismo género. Con umbral de 1 lector el diccionario llegaría a 2.916
entradas y 78,6% de cobertura, pero cada entrada se apoyaría en una sola persona, que
no es consenso sino anécdota.

Resultado: 596 nombres (258 M, 338 F) que cubren el 71,1% de los lectores sin
género declarado. El resto queda en "Desconocido", que es una categoría legítima.

Este archivo es DATOS, no lógica: no se edita a mano, se regenera.
"""

GENERO_POR_NOMBRE = {
    "aarón": "M", "abel": "M", "abraham": "M", "ada": "F", "adolfo": "M", "adrian": "M",
    "adriana": "F", "adrià": "M", "adrián": "M", "agostina": "F", "agus": "M", "agustin": "M",
    "aida": "F", "aina": "F", "ainhoa": "F", "ainoa": "F", "aitor": "M", "alba": "F", "albert":
    "M", "alberto": "M", "aldo": "M", "alejandra": "F", "alejandro": "M", "alejo": "M",
    "alessandra": "F", "alex": "M", "alexander": "M", "alexandra": "F", "alexis": "M",
    "alfonso": "M", "alfredo": "M", "alicia": "F", "alma": "F", "almudena": "F", "alonso": "M",
    "alvaro": "M", "amaia": "F", "amalia": "F", "amelia": "F", "amparo": "F", "ana": "F",
    "anabel": "F", "anastasia": "F", "andrea": "F", "andres": "M", "andrés": "M", "angel": "M",
    "angela": "F", "angeles": "F", "angie": "F", "angélica": "F", "anibal": "M", "anita": "F",
    "anna": "F", "annie": "F", "anto": "F", "antonia": "F", "antonio": "M", "anuska": "F",
    "arantxa": "F", "ari": "F", "ariel": "M", "armando": "M", "arnau": "M", "arsenio": "M",
    "arturo": "M", "asun": "F", "aureliano": "M", "aurelio": "M", "axel": "M", "ayelen": "F",
    "ayelén": "F", "bastian": "M", "bea": "F", "beatriz": "F", "bego": "F", "belen": "F",
    "belén": "F", "ben": "M", "bernardo": "M", "billy": "M", "blanca": "F", "borja": "M",
    "brenda": "F", "brian": "M", "bruno": "M", "bryan": "M", "bárbara": "F", "cami": "F",
    "camila": "F", "camilo": "M", "cande": "F", "carina": "F", "carla": "F", "carles": "M",
    "carlo": "M", "carlos": "M", "carlota": "F", "carmelo": "M", "carmen": "F", "carol": "F",
    "carolina": "F", "cat": "F", "cata": "F", "catalina": "F", "cecilia": "F", "celeste": "F",
    "celia": "F", "cesar": "M", "charlotte": "F", "chema": "M", "chica": "F", "chorche": "M",
    "christian": "M", "clara": "F", "claudia": "F", "concha": "F", "conchi": "F", "conor": "M",
    "constanza": "F", "coral": "F", "cristian": "M", "cristina": "F", "curro": "M", "cynthia":
    "F", "césar": "M", "damaris": "F", "dana": "F", "daniel": "M", "daniela": "F", "david": "M",
    "delfina": "F", "delia": "F", "desiree": "F", "diana": "F", "diego": "M", "dolores": "F",
    "douglas": "M", "dulce": "F", "désirée": "F", "edgar": "M", "edison": "M", "edith": "F",
    "eduardo": "M", "edwin": "M", "el": "M", "elena": "F", "eli": "F", "elisa": "F", "elisabet":
    "F", "elisabeth": "F", "eliseo": "M", "elizabeth": "F", "elvira": "F", "elías": "M",
    "emanuel": "M", "emilia": "F", "emilio": "M", "emma": "F", "encarni": "F", "enrique": "M",
    "eric": "M", "erick": "M", "erika": "F", "ernesto": "M", "erwin": "M", "esperanza": "F",
    "esteban": "M", "estefanía": "F", "estela": "F", "ester": "F", "esther": "F", "estrella":
    "F", "euge": "F", "eugenia": "F", "eugenio": "M", "eva": "F", "ezequiel": "M", "fabian":
    "M", "fabio": "M", "fabiola": "F", "fco.": "M", "fede": "M", "federico": "M", "felipe": "M",
    "felix": "M", "fermin": "M", "fernanda": "F", "fernando": "M", "fidel": "M", "fiorella":
    "F", "flavia": "F", "flor": "F", "florencia": "F", "fly": "M", "fran": "M", "francesco":
    "M", "francis": "M", "francisco": "M", "franco": "M", "frank": "M", "fátima": "F", "félix":
    "M", "gabriel": "M", "gabriela": "F", "gem": "F", "gema": "F", "gemma": "F", "genesis": "F",
    "gerard": "M", "gerardo": "M", "german": "M", "germán": "M", "gisela": "F", "glauka": "F",
    "gloria": "F", "gonzalo": "M", "gorka": "M", "graciela": "F", "greta": "F", "guadalupe":
    "F", "guillem": "M", "guillermo": "M", "gus": "M", "gustavo": "M", "hector": "M", "helena":
    "F", "henry": "M", "hernan": "M", "hilda": "F", "hiram": "M", "hugo": "M", "héctor": "M",
    "ignacio": "M", "igor": "M", "iker": "M", "ines": "F", "ingrid": "F", "inma": "F",
    "inmaculada": "F", "inés": "F", "irene": "F", "iria": "F", "irina": "F", "irving": "M",
    "isa": "F", "isaac": "M", "isabel": "F", "ismael": "M", "israel": "M", "itziar": "F",
    "iulmi": "F", "ivan": "M", "iván": "M", "iñaki": "M", "iñigo": "M", "jackeline": "F",
    "jacobo": "M", "jacqueline": "F", "jaime": "M", "janire": "F", "jannet": "F", "jaume": "M",
    "javi": "M", "javier": "M", "jean": "M", "jeisson": "M", "jenifer": "F", "jennifer": "F",
    "jessica": "F", "jesus": "M", "jesús": "M", "jhon": "M", "jim": "M", "joaquin": "M",
    "joaquín": "M", "joel": "M", "johan": "M", "johana": "F", "johanna": "F", "john": "M",
    "jonathan": "M", "jordi": "M", "jorge": "M", "jose": "M", "joseba": "M", "joseca": "M",
    "josefa": "F", "josep": "M", "josh": "M", "josué": "M", "josé": "M", "jota": "M", "jp": "M",
    "juan": "M", "juana": "F", "juanjo": "M", "judit": "F", "judith": "F", "julia": "F",
    "julian": "M", "julieta": "F", "julio": "M", "june": "F", "karen": "F", "karin": "F",
    "karina": "F", "karla": "F", "karlos": "M", "karol": "F", "katherine": "F", "keef": "M",
    "kevin": "M", "kike": "M", "la": "F", "lady": "F", "laia": "F", "lara": "F", "lau": "F",
    "laura": "F", "lautaro": "M", "lectora": "F", "leia": "F", "leonardo": "M", "leslie": "F",
    "leti": "F", "leticia": "F", "libros": "M", "lidia": "F", "lili": "F", "lilian": "F",
    "liliana": "F", "lily": "F", "lina": "F", "linda": "F", "lizeth": "F", "lola": "F", "loles":
    "F", "loli": "F", "loly": "F", "lore": "F", "lorena": "F", "lorenzo": "M", "lourdes": "F",
    "lucas": "M", "lucia": "F", "luciana": "F", "luciano": "M", "lucila": "F", "lucy": "F",
    "lucía": "F", "ludmila": "F", "luis": "M", "luisa": "F", "luna": "F", "luz": "F",
    "macarena": "F", "maialen": "F", "maika": "F", "maite": "F", "manoli": "F", "manolo": "M",
    "manu": "M", "manuel": "M", "manuela": "F", "mar": "F", "mara": "F", "marc": "M", "marcela":
    "F", "marcelo": "M", "marco": "M", "marcos": "M", "marga": "F", "margarita": "F", "mari":
    "F", "maria": "F", "mariajo": "F", "marian": "F", "mariana": "F", "mariano": "M", "maribel":
    "F", "maricela": "F", "mariel": "F", "mariela": "F", "marimar": "F", "marina": "F", "mario":
    "M", "marisa": "F", "marisol": "F", "mark": "M", "marta": "F", "martin": "M", "martina":
    "F", "martín": "M", "maru": "F", "mary": "F", "maría": "F", "massiel": "F", "mati": "F",
    "matias": "M", "matilde": "F", "matías": "M", "mauricio": "M", "mauro": "M", "maximiliano":
    "M", "mayra": "F", "mayte": "F", "mei": "F", "melanie": "F", "melina": "F", "melisa": "F",
    "mercedes": "F", "merche": "F", "meri": "F", "meritxell": "F", "mi": "F", "micaela": "F",
    "michael": "M", "michelle": "F", "miguel": "M", "mike": "M", "mila": "F", "milena": "F",
    "mimi": "F", "mina": "F", "mireia": "F", "miren": "F", "miriam": "F", "mirian": "F", "mj":
    "F", "moises": "M", "moisés": "M", "monica": "F", "montse": "F", "montserrat": "F",
    "morgana": "F", "mr.": "M", "mª": "F", "mónica": "F", "nacho": "M", "nadia": "F", "naiara":
    "F", "nando": "M", "nat": "F", "natalia": "F", "nataly": "F", "nati": "F", "natividad": "F",
    "naty": "F", "nazaret": "F", "nerea": "F", "nestor": "M", "nicol": "F", "nicolas": "M",
    "nicole": "F", "nicolás": "M", "nieves": "F", "noelia": "F", "noemi": "F", "noemí": "F",
    "nora": "F", "nuria": "F", "néstor": "M", "núria": "F", "olga": "F", "oliver": "M", "omar":
    "M", "orlando": "M", "oscar": "M", "osvaldo": "M", "pablo": "M", "paco": "M", "paloma": "F",
    "pam": "F", "pamela": "F", "pao": "F", "paola": "F", "paqui": "F", "patri": "F", "patricia":
    "F", "paul": "M", "paula": "F", "pedro": "M", "pepe": "M", "pia": "F", "pilar": "F", "pol":
    "M", "rachel": "F", "rafa": "M", "rafael": "M", "ramiro": "M", "ramon": "M", "ramón": "M",
    "raquel": "F", "raul": "M", "raúl": "M", "rebeca": "F", "rebecca": "F", "reyes": "F",
    "ricardo": "M", "richard": "M", "ritchie": "M", "ro": "F", "rober": "M", "roberto": "M",
    "robinson": "M", "rocio": "F", "rocío": "F", "rodolfo": "M", "rodrigo": "M", "romina": "F",
    "ronald": "M", "rosa": "F", "rosana": "F", "rosario": "F", "rose": "F", "rosita": "F",
    "rousely": "F", "roxana": "F", "ruben": "M", "rubén": "M", "runa": "F", "ruth": "F",
    "salva": "M", "salvador": "M", "sam": "M", "samuel": "M", "sandra": "F", "sandro": "M",
    "sandy": "F", "santi": "M", "santiago": "M", "sara": "F", "sarah": "F", "saúl": "M",
    "sebastian": "M", "sebastián": "M", "sergi": "M", "sergio": "M", "señor": "M", "sharon":
    "F", "sheila": "F", "silvia": "F", "simón": "M", "sine": "F", "sir": "M", "sofía": "F",
    "sol": "F", "soledad": "F", "sonia": "F", "sophie": "F", "steven": "M", "susana": "F",
    "susi": "F", "tamara": "F", "tania": "F", "tere": "F", "teresa": "F", "thais": "F", "tinta":
    "F", "tomas": "M", "tomás": "M", "toñi": "F", "toño": "M", "unai": "M", "uriel": "M",
    "valentina": "F", "valentín": "M", "valeria": "F", "vane": "F", "vanesa": "F", "vanessa":
    "F", "vero": "F", "veronica": "F", "vicent": "M", "vicente": "M", "victor": "M", "victoria":
    "F", "violeta": "F", "virginia": "F", "vivian": "F", "viviana": "F", "víctor": "M",
    "walter": "M", "wendy": "F", "xavi": "M", "xavier": "M", "ximena": "F", "yamil": "M",
    "yayo": "M", "yessica": "F", "yolanda": "F", "álvaro": "M", "ángel": "M", "ángela": "F",
    "óscar": "M"
}
