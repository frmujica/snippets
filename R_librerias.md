### Libreria: "dplyr"

Esta libreria pretenece al conjunto de librerias de CRAW

Contiene utilizades para realizar las siguiente operaciones sobre una DataFrame:

```R

# Seleccinar campos
select(dataFrame, "campo1", "Campo2", CampoN")

# Filtrar datos
filter(dataFrame, "campo1 = 'ABC' ")

# Ordnar
arrange(Data_Frame, "campo1"...)

# Añadir campos calculados
mutate(DataFrame, Campo_numevo = "")

# Aplicar operaciones
summarize(DataFrame, sum(campo1)

## Concatenar operaciones : %>%
# la salida de una operacion es la entrada de la siguiente

```

### Libreria: "R.utils"

Utilizada para acceder a archivos comprimidos

```R

bunZip2(f1, f2,,,

ó 

format.object.size(fileinfo("f1")$size, "auto)
```

### Libreria: "rvest"

Utilizada para leer codigo HTML

```R

read_html("http://...")

# Me quedo con los nodos tipo links
v1 = html_nodes(var_html, "a")

# de los links extraidos, me quedo con la lista de atributos href
v2 = html_attr(v1, "href")
```

### Libreria: "string"

Utilizada para el tratamien de cadenas de textos

```R

str_subset(V2, "\\.bz2")

```

### Libreria: "readr"

Utilizada para cargar ficheros

```R

# Leemos un ficheros
f <- read_csv("f1", progress=T)

# obtenemos su tamaño
object.size(f)

```


### Libreria: "data.table"

Permite cargar ficheros y usar un tratmaiento especial para DataFrames

```R
# Lectura de fichero
fread("f1")
```


### Libreria: "data.table"

Permiteejecutar varios hilos o multi-tarea

```R
registerDoParalel(cores=detectCores()-1)
```

### Libreria: "sql.df"

Utilizada para atactar a bases de datos

```R
v1 <- sqldf("Select * From Tabla")

# o a ficheros

# Cargamos el fichero
v1 <- read_csv("files1")

# Lanzamos una select sobre el fichero
v2 = sqldf("select * from v1")

```


### Libreria: "DBI"

Ataque a bases de datos

```R

v1 <- dbiConnect(RSQLITE__SQLite(), path=":memory")

tmp <- timefile()

con <- dbconnect(RSQLite::SQLite, tmp)

dbwriteTable(con, "table", data_frame)

dbListaTables(con)

dbFields(con, "table")

v1 -<- dbSendQuery(con, "select * from ...")
var <- dbFech(v1)
dbClearReset(var)

dbDisconnect(con)
UnLink(tmp)

```

### Libreria: "ggplot

Utilizada para generar graficos







