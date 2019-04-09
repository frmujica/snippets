
## Bagging ( Combinacion de modelos en paralelo )

Entrenmaos el mismo modelo con caracteristicas diferentes de forma aleatoria

```python

# Ejemplo de una Regresion Logistica

# Importamos la liberia
from sklearn.ensemble import BaggingClassifier

# instanciamos la libreria, indicando que tipo d emodelo vamos a usar 
# e indicando n_estamators que nis indica le numero de modelos y variadades a probar
clf = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=100, , oob_score=3)

# Vemos el resultado del modelo con un crossvalidatios aplicando una metrica accuracy y viendo la media
cross_val_score(clf,X,y,scoring="accuracy").mean()


# Ejemplo para k-vecinos

# Cargamos la libreria
from sklearn.ensemble import BaggingClassifier

# Creamos la combinacion
clf=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=4),n_estimators=100,oob_score=True)

# Entrenamos el modelo
clf.fit(X,y)

# Le pasamos una metrica accuracy
cross_val_score(clf,X,y,scoring="accuracy").mean()


# Ejemplo con Randomfores

# Load the library
from sklearn.ensemble import RandomForestClassifier

# Create an instance
clf = RandomForestClassifier(max_depth=4)

# Fit the data
clf.fit(X,y)

```


## Gradient Boosting

```python

# Cargamos la libreria
from sklearn.ensemble import GradientBoostingClassifier

# Creamos el combinador de modelos

clf = GradientBoostingClassifier(max_depth=4)

# ó segun queremos usar clasificador o reresor 
reg = GradientBoostingRegressor(max_depth=4, n_estimators=100,learning_rate=0.1)

# entrenamoento de los datos
clf.fit(X,y)


# Aplicamos metrica con Cross Vlidation 
from sklearn.model_selection import cross_val_score
cross_val_score(reg,X,y,scoring="neg_mean_absolute_error").mean()

# aplicamos el modelo buscando los mejores parameotros con GridSearchCV

# Cargamos la libreria
from sklearn.model_selection import GridSearchCV

# Creamos el modelo
reg = GridSearchCV(GradientBoostingRegressor(n_estimators=50),
                  param_grid={"max_depth":np.arange(2,10),
                             "learning_rate":np.arange(1,10)/10},
                  scoring="neg_mean_absolute_error",
                  cv=5)

# Entrenamos el modelo
reg.fit(X,y)
                  

```
