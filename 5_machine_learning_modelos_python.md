# Machine Learning - Aprendizaje supermivsado 

```
 Tecnica para deducir una funcion a partir de los elementos entrenados.
  
 La finalidad es crear un a funcion capaz de predecir un valor desde ejemplo dados.
  
 Requiere tener datos de entrenamientos etiquetados

           - Una entrada puede ser un vector con caractereisticas

           - la salida es la etiqueta de estas caractereisticas
 
 Tipos:

 Regresion     :: si la salida es un numeros (regresion linea... . MAE, correlacion y bias) (unimos a correlacion)
 Clasificacion :: si la salica es una clase ( regresion ligistica, vecinos, lector de soportes y arboles de decision )
 
 Metricas      ::  sistemas para evaluar lo bueno que es el modelo
```

```python

# Ejmeplo de clasificacion, modelo para REGRESION LOGISTICA

Ej = glm(data= My_Dataframe,
         family = "binomial",
         formula = var_a predecir~variable_predictora_1,variable_predictora_2...variable_predictora_N,
         wheight = var1, var2...)

```


## =============
## GridSearchCV
## =============

Antes de comenza con los modelos, vamos a ver las funcines de <b>GridSearchCV</b>, al que podemos darle un modelo, y un rango de parametros para dicho modelo y se encarga de devolvernos la mejor combinacion de parametros para nuestro modelo, en funion del los rangos facilitados.

En resumen, las funciones <b>GridSearchCV</b>, la utilizaremos para calcular automaticamente los mejores parametros de un modelo dado.

```python

# EJEMPLO de GridSearchCV para un modelo "KNeighbor"

# Instanciamos las funciones de GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instanciamos el modelo KNeighbor, en este caso para realizar runa regresión
from sklearn.neighbors import KNeighborsRegressor

# Configuramos nuestro GridSearchCV, indicandole que modelo es el que queremos evaluar
# y el rango de parametros que queremos que evalue para que nos de la mejor combinacion de parametros.

# En este caso, indicamos que el rango de vecinos mas cerfcanos a buscar sea entre 3 y 50 vecinos o datos mas cercanos.
reg_test = GridSearchCV(KNeighborsRegressor(),
                        param_grid={"n_neighbors":np.arange(3,50)}
                        )

# Vamos a entrenar el modelo o lanzar nuestro GridSearchCV, para que encuentre nuestros mejores parametros 
# para nuestro modelo KNeighborsRegressor, usando todas las combinaciones segun los parametros indicados.
reg_test.fit(X,y)

# Vamos a extraer de nuestro entrenamiento los mejores parametros encontrados para nuestro modelo KNeighborsRegressor
reg_test.best_params_ 

# Vamos a obtener el mejor resultado obtenido
reg_test.best_score_ 

# Me devuelve los mejores estimadores.
reg.best_estimator_

```




## ===========
## Regression
## ===========

### Linear Regression (Regression)

```python

# Cargamos la libreria para realizar un regrersion logistica
from sklearn.linear_model import LinearRegression

# Create an instance of the modelo
reg = LinearRegression()

# Entrenamos el modelo, pasando como parametro, una matriz de nuestras variables de entrenamiento
# y un vector con nuestra variable a predecir.
reg.fit(X,y)

# Podemos realizar una prediccion, pasando como parametro una matriz de variables
# similares a nuestra matriz X
reg.predict([[2540],[3500],[4000]])
```

### KNeighborsRegressor (Regression)

Este modelo lo podemos usar tanto pra regresion como para clasificacion

En este ejeplo lo usaremos lara realizar una regresión.

(Debemos indicar los puntos de corte o de decision en el caso de usarar para clasificacion)

Busca distancias entre elemnos, y debemos indicar hasta cuantos vecinos puede buscar.

Se basa en los angulos o cosenos de los datos para buscar los vecinos mas cercanos.

<li>n_neighbors   :: numero de vecinos a buscar mas cercanos</li>

```python

# Cargamos la libreia KNeighborsRegressor desde sklearn
from sklearn.neighbors import KNeighborsRegressor

# Instanciamos la clase y la inicializamos indicando cuantos vecinos debe de buscar
# n_neighbors=2   : numero de vecinos a buscar durante el entrenamiento del modelo.
regk = KNeighborsRegressor(n_neighbors=2)

# Entrenamos el modelo, para ello, pasamos como parametro nuestra matri de variables predictoras
# y un vector y con las variables a predecir.
regk.fit(X,y)

```

``` Version con lenguaje R

# K : Cuantos vecinos ebemos de buscar,

Knn(DataFrame_Training,
   Data_Frame_Test,
   cl,
   k=5,
   prob=T
   )


```

### Decision Tree - Arboles de decision (Regression)

Este modelo necesita como minimo los parametros de:

<li>Max_depth        :: Number of Splits o numero de saltos o profundidad</li>
<li>Min_samples_leaf :: Minimum number of observations per leaf o numero inimo de observaciones por salto</li>

```python

# Opcion 1

# Importmos la libreria
from sklearn.tree import DecisionTreeRegressor

# Instanciamos el objeto e inicializamos el nuero de saltos o profuncidad de nuestro arbon de decisioon
regd = DecisionTreeRegressor(max_depth=3)

# Entrenamos nuestro modelo, pasando como paramtro una matriz de varialbles predictoras 
# y un vecto y con los datos a predecir.
regd.fit(X,y)


# Vamos a repetir el ejemplo utilizando las funciones de GridSearch 

# Instancimos las funciones de GridSearchCV
from sklearn.model_selection import GridSearchCV

# Creamos un objeto  GridSearchCV pasando como parametros:
# El modelo del que queremos conocer los mejores paramtros
# Y en este caso
# max_depth : con el rango de posible profundidades que puede tener nuestro arbol.
# min_samples_leaf : y el rango minimo de muestras por prodfundidad que puede tener nuestro arbol de decision
reg = GridSearchCV(DecisionTreeRegressor(),
                  param_grid={"max_depth":np.arange(2,8),
                              "min_samples_leaf":[10,30,50,100]},
                  cv=5,
                  scoring=make_scorer(corr_test))
                  
# entrenamos nuestra funcion GridSearchCV, pasando como parametro un matriz con nuestras variables predictoras
# y un vector con nuestras variables a predecir.
reg.fit(X,y)

# La libreria export_graphviz, la usaremos para visualizar nuestro arbol de decision
# Método que permite exportar los resultados de un árbol de decisión al formato DOT de Graphviz.
# importaremos la libreria desde sklearn
from sklearn.tree import export_graphviz

# Importamos libreria que recibe un objeto export_graphviz y puede intarlo. 
import pydotplus

# importamos la librria para tratar Strings
from string import String

# importamos lobrria para tratar ficheros
import io

# Instanaimos un objeto que nos permite tratar ficheros
dot_data = io.StringIO()

# Exportamos nuestro arbol de decision con la funcion export_graphviz
# pasando como paramtros
# El mejor modelo obtenido desde nuestro GridSerachCV
# El objeto fichero
# ...
export_graphviz(reg.best_estimator_, 
                out_file=dot_data,
                filled=True, 
                rounded=True,
                special_characters=True)

# Guardamos el objeto
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Importamos una libreria que nos permite mostrar imagenes
from IPython.display import Image

# Llamamos la funcion Image de este libreria pasando como parametro el grafico de nuestro madelo transformado en imagen (graph.create_png())
Image(graph.create_png())

# Pintamos el grafico
plt.plot(xgrid,reg.best_estimator_.predict(xgrid),color="red")
plt.scatter(X,y);

```


### RandomForest (Regression)

Este modelo se puede definir como un grupo de modelos "Decision Tree"


Los parametros basicos que necesita son:

<li>Max_depth        :: Number of Splits, o profundidad</li>
<li>Min_samples_leaf :: Minimum number of observations per leaf, o numero de muestras por profundidad</li>

```python
Pra el ejemplo y buscar los mejores paramttros de nuestro modelo, vamos a usar las funciones de GridSearchCV

# RandomizedSearchCV -> es como el GridSearchCV pero prueba n_iter=5 combinaciones de forma aleatoria

# Importamos la libreria RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# instanaciamos e inicializamos nuestro objeto RandomizedSearchCV
# Pasamos como parametro:
# DecisionTreeRegressor()            : El modelo a evluar
# max_depth":np.arange(2,8)          : rando de profuncidades
# min_samples_leaf":[10,30,40,60]    : rango d datos a tratar en cada nivel
# cv=5                               : 
# scoring="neg_mean_absolute_error"  : Metrica con la mque medir el errordurante el apredizaje
# n_iter=5                           : numero de combinaciones a probar
                   
rs = RandomizedSearchCV(DecisionTreeRegressor(),
                   param_distributions={"max_depth":np.arange(2,8),
                                        "min_samples_leaf":[10,30,40,60]
                                       },
                   cv=5,
                   scoring="neg_mean_absolute_error",
                   n_iter=5
                  )

```


## ===============
## Classification
## ===============

### Logisitc Regression (Classification)

```python

# Importamos la libreria
from sklearn.linear_model import LogisticRegression

# Creamos una instancia del objeto
clf=LogisticRegression()

# Entrenamos el modelo, pasando como parametros una matriz con las variables predictoras y un vector con la variable a predecir.
clf.fit(X,y)

```


### k-Nearest Neighbors o K-vecinos (KNN) (Classification)

Anque este módelo se puede utilizar tanto para regresión como clasificación, se suele utilizar mas para clasificación.

Para realizar una predicción normalmnete son fijamos en tres parametros fundamentales:

- Facilidad para interpresar la salida
- Tiempo de calculo
- Poder predictivo

<table>
<tr>
<td></td>
<td>Regresion Logistica</td>
<td>CART</td>
<td>Random Forest</td>
<td>KNN</td>
<tr>
<tr>
<td>Facilidad para interpresar la salida</td>
<td>2</td>
<td>3</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>Tiempo de calculo</td>
<td>3</td>
<td>2</td>
<td>1</td>
<td>3</td>
</tr>
<tr>
<td>Poder predictivo</td>
<td>2</td>
<td>2</td>
<td>3</td>
<td>2</td>
</tr>
</table>
                                          

Este módelo se suele usar por su facil interpretación y bajo tiempo de calculo.

Parametros
<li>n_neighbors :: Numero de vecinos a buscar mas cercanos</li>

Este parmetro influlle en las metricas de rror. Tener en cuenta.

```python

# Importamos la libreria para poder usar el modelo NKK
from sklearn.neighbors import KNeighborsClassifier

# Instanciamos el objeto de nuestro modelo, indicando como parametro el numeor de vecinos a localizar
regk = KNeighborsClassifier(n_neighbors=2)

# Entrenamos nuestro modelo, pasando como parametro una matris con a lista de variables predictoras y un vector con la variable a predecir.
regk.fit(X,y)
```

### Navie Bayes

El algoritmo Naive Bayes es un método intuitivo que utiliza las probabilidades de cada atributo que pertenece a cada clase para hacer una predicción. Es el enfoque de aprendizaje supervisado que se le ocurriría si quisiera modelar un problema de modelado predictivo de forma probabilística.

Simplifican el cálculo de probabilidades al suponer que la probabilidad de que cada atributo pertenezca a un valor de clase dado es independiente de todos los demás atributo

contrar del módelo:

* Requerir a eliminar características correlacionadas Porque son votados dos veces en el modelo y puede llevar a inflando importancia.

* Si una variable categórica tiene una categoría en el conjunto de datos de prueba que no se observó en el conjunto de datos de entrenamiento , entonces el modelo asignará una probabilidad cero . No podrá hacer una predicción.

* Este tipo de modelos no permite que las varaibles predictoras interactuen entre si. Por lo que se suele usar un modelo de "arboles"


```PYTHON

# Importamos la libreria para poder usar el módelo
From sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Instanciamos el objeto de nuestro modelo
gnb = GaussianNB ()

# Creamos un vector con las variables predictoras
used_features = [
    " Clase " ,
    " Sex_cleaned " ,
    " Edad " ,
    " SibSp " ,
    " Parch " ,
    " Tarifa " ,
    " Embarked_cleaned "
]

# Entrenamos nuestro modelo pasandole como parametro una matriz con nuestras varaibles predictoras y un vector con nuestra variable a predecir
gnb.fit(
    X_train [used_features] .values,
    X_train [ "Sobrevivio" ]
)

# Probamos a realizar una predicción
y_pred = gnb.predict (X_test [used_features])


```

```R

# Modelo de clasificacion de la librería "e1071"

# Importacion de la libreria
library (e1071)

# Creacion del modelo
nb = navieBayes(var_a_predecir~var_predictora_1+var_predictora_2...var_predictora_N, data=DataFrame)

# informacion del modelo
Summary(nb)

# Prediccion 
predict(nb, newdata=New_Data_Frame)

# Visualizacion de los datos en un grafico
hist( predict( modelo, newdata=New_Data_Frame, type="responsive", brear=10)

```


### Decision Tree (Classification)

Pros: 

* Este modelo permite la interactuacion entre las varaibles predictoras.

* Es fácil de usar

* Metodo muy fiable 

Contras:

* Pero es demasiado flexible


Podemos indicar el punto de corte o decisorio, y que se suele indicar con con puntos que mas separen los datos.

Conceptos:

Nodo        :: Punto de decision
Hoja        :: Lo que cuelga del nogo
Profundidad :: Numero de decisiones

Parametros:
<li>Max_depth         :: Numero de diviones</li>
<li>Min_samples_leaf  :: Numro minimo de hojas</li>
<li>max_depth         :: Profundidad</li>

```python

# Importamos la libreria para poder usar nuestro modelo
from sklearn.tree import DecisionTreeClassifier

# Instanciamos el objeto de nuestro modelo. durante la instanacia podemos especificar algunos de los parametros
clf = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)

# Entrenamos nuestro modelo, pasando como parametros una matris con las variables predictoras y un vector con la variable a predecir.
clf.fit(X,y)


# Podemos utilizar las funciones de GridSearchCV para encotnrar de entre un rango de parametros, la mejor
# parametrización d enuestro modelo. Vemos un ejemplo:

# Ejemplo con GridSearchCV

# Importamos la libraria para poder usar nuestro modelo
from sklearn.tree import DecisionTreeClassifier

# Importamos la libreria para usar las funciones de GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instanciamos el objeto GridSearchCV y lo parametrizamos
# DecisionTreeClassifier()                  : modelo a evaluar
# "max_depth":np.arange(1,100,20)           : Rango de profuncidades de nuestro arbol de decisión
# "min_samples_leaf":np.arange(20,100,20)   : rango de hojas
# cv                                        : numero de validaciones cruzadas que va a realizar
# scoring                                   : reglas de evaluacion del modelo o metrica con el que se mide nuestro modelo
clft = GridSearchCV(DecisionTreeClassifier(),
                    param_grid={"max_depth":np.arange(1,100,20),
                                "min_samples_leaf":np.arange(20,100,20)
                               },
                    cv=5,
                    scoring="precision"
                  )

# Entrenamos el modelo
clft.fit(X,y)

# Nos devulve la instruccion con cumnplimentada con los parametros con mejor resultado
clft.best_estimator_

# Nos devulve los mejores parametros con los que ha obtenido el mejor resultado
clft.best_params_

# mejor resultado
clft.best_score_

```

Ejemplo en R

```R

# importamos la libreria
library (tree)

# Creamos el modelo
# Con el signo "+" añadimos la variable predictora
# Con el signo "*" ademas indicamos que las varaibles interactuan entre ellas.
modelo = tree(variable_a_predecir~Variable_predictora_1+Variable_predictora_2...Variable_predictora_N, Data_Frame)

# prediccion con los datos nuevos
predict(modelo, newdata=New_Data_Frame)

# Dibujamos en pantallas
plot(modelo, Y=Null, type=C("____", "_____"))
```


## RamdomFores (Classification)

Este modelo generar muchos arboles de decision independientes construidos a partir de datos ligeramente distintos

Contra: 

* Estemodelo no admite valores NAN

Parametros
<li>n_estimator :: numero de arboles</li>
<li>max_depth   :: profunciondad</li>
<li>min_samples :: numero de testeos</li>
<li>obs=-1      ::  Procesadores a utilizar</li>

```python

# Importamos la libreria
from sklearn.ensemble import RandomForestClassifier

# Creamos el modelo
# Parametros:
# max_depth=3            : Profundidad de nuestros arboles
# min_samples_leaf=20    : Numero de testeos minimos
# n_estimators=100       : Número de arboles
# n_jobs=-1              : Numero d eprocesador a utilizar (-1 TODOS)
clf = RandomForestClassifier(max_depth=3, min_samples_leaf=20, n_estimators=100, n_jobs=-1)

# Si deseamos realizar una metrica para lo bueno que es nuestro modelo, podemos pusar usar la metrica CrossValidaton
# Pasando como parametros, nuestro modelo
# la patriz de train
y el vector de train
cross_val_score(clf, X,y).mean()

# Entrenamiento pasando como parametro la matriz de varialbes predictoras y un vector con nuestra varaiable a predecir.
clf.fit(X,y)
```

```R
randomFirest(variable_a_predecir~variable_predictora_1+variable_predictora_2...variable_predictora_N, DataFrame)
```


### Support Vercor (SVC) (Classification)

Separa los puntos de una clase de otra clase de forma linea.

El objetivo de un SVC lineal es ajustarse a los datos que proporcione, devolviendo un hiperplano de "mejor ajuste" que divide o clasifica sus datos

Parametros
<li>kernel="linear" :: Para indicar que tipo de separacion valor a utilizar</li>
<li>C=10            :: Para indicar cuantos puntos podemos dejar que se crecuen al otro lado de la linea</li>


```python
# Importamos la libreria de nuestro modelo
from sklearn.svm import SVC

# Instiamos nuestro modelo
# Parametros
# kerlen = añade dimensiones
# kernel="linear"    : Tipo de separacion o division d elos datos queremos hacer
# C=10               : Indicmaos cuantos puntos muden cruzarse al lado de linea que no debía o como de permisivo es nuestro modelo
clf = SVC(kernel="linear", C=10)

# Metria de nuestro modelo 
# Parametros
# clf      : numestro modelo
# X        : Matriz con nuestras variable predicoras
# y        : Vector con nuestra variable a predecir
cross_val_score(clf,X,y)

# Entranmiento del modelo pasando como parametro una matriz con las varialbes predicroras y un vector con nuestra varaible a predecir
clf.fit(X,y)

# Extracción d ela media de nuestrsa metrica
cross_val_score(clf,X,y).mean()


# Podemos tambien buscar, dado un rango de parametros, la mejor parametrizacion de nuestro modelo SVC

# Instanciamos las funciones de GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instaciamos nuestro objeto GridSearchCV y lo parametrizamos
# Parametros:
# SVC(kernel="poly", degree=10)    :  modelo a evaluar indicando que tipo de clasifiacionen quiero
#     kernel="poly" : Mientras "lineal" busca un hiperplano linal, poly busca hiperparametros no lineales
#     degree10: es un parámetro usado cuando el núcleo está configurado como 'poli'. 
#               Es básicamente el grado del polinomio utilizado para encontrar el hiperplano para dividir
#               los datos.
# cv=5                : numero de validaciones cruzadas que va a realizar
# scoring="accuracy"  : Metria con el el evaluar el modelo para extraer la mejor parametrizacion
clf = GridSearchCV(SVC(kernel="poly", degree=10), 
                  param_grid={"C":[1,10,100,1000],
                              "degree":[2,3,4]
                             },
                   cv=5,
                   scoring="accuracy"
                  )
                  
# Entrenamos los datos pasando como parametros una matriz con las varialbes predictoras y un vector con la variable a predecir
clf.fit(X,y) # entrenamos los datos

# Devulve la generacion del modolo con los parametros que mejor resultado han dado
clf.best_estimator_

# Nos devulve los mejores parametros encontrdos
clf.best_params_

# Nos devulve el mejor resultado encontrado con los paramentros encontrados
clf.best_score_

# Probamos una prediccion
clf.predict(X_test)

```
