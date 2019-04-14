# Machine Learning Snippets

```
 Machin Learning - Aprendizaje supermivsado 
 
 Tecnica para deducir una funcion a partir de los elelentos entrenados
 
 - El resultado puede ser un valor numerico concreto (REGRESION) (lm)
 - O puede devolver una etiqueta (CLASIFICACION) (glm)
 
 La finalidad es creara un a funcion capaz de predecir un valor desde ejemplo dadods.
  
 Requiere tener datos de entrenamientos etiquetados

           - Una entrada puede ser un vector con caractereisticas

           - la salida es la etiqueta de estas caractereisticas
 
 Tipos:

 Regresion     :: si la salida es un numeros (regresion linea... . MAE, correlacion y bias) (unimos a correlacion)
 Clasificacion :: si la salica es una clase ( regresion ligistica, vecinos, lector de soportes y arboles de decision )
 
 Metricas      ::  Para evaluar lo bueno que es el modelo
```

```python

# Ejmplo de clasificacion

Ej = glm(data= My_Dataframe,
         family = "binomial",
         formula = var_a predecir~variable_predictora_1,variable_predictora_2...variable_predictora_N,
         wheight = var1, var2...)
         

```



## GridSearchCV

```python

# EJEMPLO de GridSearchCV para PARA KA-VECINOS

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor

reg_test = GridSearchCV(KNeighborsRegressor(),
                        param_grid={"n_neighbors":np.arange(3,50)}
                        )

# Fit will test all of the combinations
reg_test.fit(X,y)

# me indica el numero de vecinos que menos errores
reg_test.best_params_ 

# me devulve el mejor 
reg_test.best_score_ 

# Me devuelve el mejor valor
reg.best_estimator_

```

## RamdomFores

Parametros
<li>n_estimator :: numero de arboles</li>
<li>max_depth   :: profunciondad</li>
<li>min_samples :: numero de testeos</li>
<li>obs=-1      ::  Procesadores a utilizar</li>

```python

# Importamos la libreria
from sklearn.ensemble import RandomForestClassifier

# Creamos el modelo
clf = RandomForestClassifier(max_depth=3, min_samples_leaf=20, n_estimators=100, n_jobs=-1)

# pasamos la metrica CrossValidaton
cross_val_score(clf, X,y).mean()

# Entrenamiento
clf.fit(X,y)
```


## ===========
## Regression
## ===========

### Linear Regression (Regression)

```python

# Load the library
from sklearn.linear_model import LinearRegression

# Create an instance of the model
reg = LinearRegression()

# Fit the regressor
reg.fit(X,y)

# Do predictions
reg.predict([[2540],[3500],[4000]])
```

### k nearest neighbor (Regression)

<li>n_neighbors   :: numero de vecinos a buscar mas cercanos</li>

```python

# Load the library
from sklearn.neighbors import KNeighborsRegressor

# Create an instance
regk = KNeighborsRegressor(n_neighbors=2)

# Fit the data
regk.fit(X,y)

```

### Decision Tree - Arboles de decision (Regression)

Parametros
<li>Max_depth        :: Number of Splits</li>
<li>Min_samples_leaf :: Minimum number of observations per leaf</li>

```python

# Opcion 1

# Load the library
from sklearn.tree import DecisionTreeRegressor

# Create an instance
regd = DecisionTreeRegressor(max_depth=3)

# Fit the data
regd.fit(X,y)


# Visualizacion de una arbol de decision desde un GridSearch

reg = GridSearchCV(DecisionTreeRegressor(),
                  param_grid={"max_depth":np.arange(2,8),
                              "min_samples_leaf":[10,30,50,100]},
                  cv=5,
                  scoring=make_scorer(corr_test))
                  
reg.fit(X,y)

from sklearn.tree import export_graphviz

import pydotplus
from string import String

import io
dot_data = io.StringIO()

export_graphviz(reg.best_estimator_, 
                out_file=dot_data,
                filled=True, 
                rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

from IPython.display import Image
Image(graph.create_png())

# Pintamos el grafico
plt.plot(xgrid,reg.best_estimator_.predict(xgrid),color="red")
plt.scatter(X,y);

```


### RandomForest (Regression)

Parametros
<li>Max_depth        :: Number of Splits</li>
<li>Min_samples_leaf :: Minimum number of observations per leaf</li>

```python

# RandomizedSearchCV -> es como el GridSearchCV pero prueba n_iter=5 combinaciones de forma aleatoria

from sklearn.model_selection import RandomizedSearchCV

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

# Load the library
from sklearn.linear_model import LogisticRegression

# Create an instance of the classifier
clf=LogisticRegression()

# Fit the data
clf.fit(X,y)

```


### k nearest neighbor (Classification)

Parametros
<li>n_neighbors :: Numero de vecinos a bus r mas cercanos</li>

```python

# Load the library
from sklearn.neighbors import KNeighborsClassifier

# Create an instance
regk = KNeighborsClassifier(n_neighbors=2)

# Fit the data
regk.fit(X,y)
```


### Decision Tree (Classification)

Parametros:
<li>Max_depth         :: Numero de diviones</li>
<li>Min_samples_leaf  :: Numro minimo de hojas</li>
<li>max_depth         :: Profundidad</li>

```python

# Import Library
from sklearn.tree import DecisionTreeClassifier

# Create instance
clf = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)

# Fit
clf.fit(X,y)


# Ejemplo con GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

clft = GridSearchCV(DecisionTreeClassifier(),
                    param_grid={"max_depth":np.arange(1,100,20),
                                "min_samples_leaf":np.arange(20,100,20)
                               },
                    cv=5,
                    scoring="precision"
                  )

# Crgamos el modelo
clft.fit(X,y)

# Nos devulve la instruccion con cunmplimnetada con los parametros con mejor resultado
clft.best_estimator_

#Nos devulve los mejores parametros con los que ha obtenido el mejor resultado
clft.best_params_

# mejor resultado
clft.best_score_

```



### Support Vercor (SVC) (Classification)

Separa los puntos de una clase de otra clase de forma linea.

Parametros
<li>kernel="linear" :: Para indicar que tipo de separacion valor a utilizar</li>
<li>C=10            :: Para indicar cuantos puntos de podemos dejar que se crecuen al otro lado de la linea</li>


```python
# Import Library
from sklearn.svm import SVC

# Create instance
# kerlen = añade dimensiones
clf = SVC(kernel="linear", C=10)

# Opcion 1: Procesamos con Cross Validation
cross_val_score(clf,X,y)

# Resultado sacando la media
cross_val_score(clf,X,y).mean()

# Opcion 2
clf = GridSearchCV(SVC(kernel="poly", degree=10), 
                  param_grid={"C":[1,10,100,1000],
                              "degree":[2,3,4]
                             },
                   cv=5,
                   scoring="accuracy"
                  )
                  
# Entrenamos los datos
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
