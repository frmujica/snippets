# CROSS VALIDATION

## Validacion cruzada

Da como resultado medidas estadísticas como la media y la desviación estandar. 
Esta validación se puede confirmar con gráficos de tipo box charts, scattered matrix e histogramas

Parametros
<li>clf                 :: Es el modelo</li>
<li>X                   :: Destino para el eje X</li>
<li>y                   :: Destino para el eje y</li>
<li>CV=5                :: Genera 5 conjuntos de test.</li>
<li>scoring='accuracy'  :: Metrica a utilizar</li>

```PYTHON

#
# Lista de scores o metricas disponibles
#
# import sklearn
# sorted(sklearn.metrics.SCORERS.keys())


from sklearn.model_selection import cross_val_score

# errores
err = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("Errores:   :", err)

# Varianza
varianza = cross_val_score(clf, X, y, cv=5, scoring='accuracy').var()
print ("Vairanza   :", varianza)

# desviacion
desviacion = cross_val_score(clf, X, y, cv=5, scoring='accuracy').std()
print ("Desviacion :", desviacion)

# media
media = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
print ("Media      :", media)


# Ejemplo de validacion cruzada a un arbol de decision

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Creamos el modelo
regd = DecisionTreeRegressor(max_depth=5)

# aplicamos validacion cruzada
cross_validate(regd, X, y, cv=5, scoring="neg_mean_absolute_error")

# Con GreadSearchCV buscamos los mjores parametros
regi = GridSearchCV(DecisionTreeRegressor(),
                   param_grid={"max_depth":np.arange(2,8)},
                   cv=5,
                   scoring="neg_mean_absolute_error"
                  )

```


## ACCURACY

Es el porcentaje total de elementos clasificados correctamente.

```PYTHON

# importamos la libera de la metrica para sacar el accuracy

from sklearn.metrics import accuracy_score
accuracy_score(y_test, clf.predict(X_test))

```

## MAE

Metria MAE (suma de los valores absolutos) y nos da el error para todas las casos con esta metrica

```PYTHON

# importamos la libera de la metrica
from sklearn.metrics import mean_absolute_error

print("Error con SKLEARN MAE: ", mean_absolute_error(y_test, pred))

print ("Podemos calcularlo a mano:", np.mean(np.abs(y_test-pred)))

# Podemos ver el histograma del resultado
plt.hist(y_test-pred, bins=50);



# Ejemplo de metrica aplicada a KNN o K-Vecinos

# cargamos la librería
from sklearn.neighbors import KNeighborsRegressor

maes=[]

for i in range(4,100):
    regK = KNeighborsRegressor(n_neighbors=i)
    regK.fit(X_train, y_train)
    maes.append(mean_absolute_error(y_test, regK.predict(X_test)))

plt.plot(maes)


```

## MAPE

Suma de todos los valores absolutos y divide entre el total de valores.

```PYTHON

# Podemos implementarlo manualmente

MAPE = np.mean(np.abs(y_test-pred)/y_test)

print("Error con MAPE:", MAPE)

# Podemos ver el histograma del resultado
plt.hist((y_test-pred)/y_test, bins=50);

```


## RMSE

La diferencia de los cuadrados de los datos en valos absoluto, de este modo se penalizan mas los errores

```PYTHON

# instanciamos la libreria 
from sklearn.metrics import mean_squared_error

# y caculamos la metrica para ver el rango de error
np.sqrt(mean_squared_error(y_test, regK.predict(X_test)))

```


## CORRELACIONAL

Busca correlacion entre la prediccion y los valores con los que realiza la prediccion

```PYTHON

# Opcion 1

def corr_test(y_test, pred):
    return np.corrcoef(y_test, pred)[0][1]

# Importamos la libreria
from sklearn.metrics import make_scorer

reg = cross_val_score(regd, X, y, cv=5, scoring=make_scorer(corr_test)).mean()


# Opcion 2

# importamos la libreria
from sklearn.model_selection import GridSearchCV

reg = GridSearchCV(DecisionTreeRegressor(),
                   param_grid={"max_depth":np.arange(2,8),
                              "min_samples_leaf":[10,30,50,100]},
                   cv=5,
                   scoring=make_scorer(corr_test)
                  )

# Entrenamos el modelo
reg.fit(X,y)

# Obtenemos los mejores parametros
regi.best_params_

```


## Bias

Es la media de los errores, lo que nos permite ver si los errores están desviados a una lado o a otro

```PYTHON

# instanciamos la libreria 
from sklearn.metrics import mean_squared_error

# y caculamos la metrica para ver el rango de error
np.sqrt(mean_squared_error(y_test, regK.predict(X_test)))

```


## Matriz de confusion

Tabla que describe el rendimiento de un modelo supervisado de Machine Learning en los datos de prueba, donde se desconocen los verdaderos valores. Se llama “matriz de confusión” porque hace que sea fácil detectar dónde el sistema está confundiendo dos clases.

True Positives (TP): cuando la clase real del punto de datos era 1 (Verdadero) y la predicha es también 1 (Verdadero)

Verdaderos Negativos (TN): cuando la clase real del punto de datos fue 0 (Falso) y el pronosticado también es 0 (Falso).

False Positives (FP): cuando la clase real del punto de datos era 0 (False) y el pronosticado es 1 (True).

False Negatives (FN): Cuando la clase real del punto de datos era 1 (Verdadero) y el valor predicho es 0 (Falso).

```PYTHON

# Opcion 1

# libreria para genera la matriz de confusion
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, clfK.predict(X_test))


# Opcion 2

# Tambine podemos pintar la matriz de consuion
import seaborn as sns

sns.heatmap(confusion_matrix(y_test, clfK.predict(X_test)) )


# Opcion 3

# Otra forma de generar la matriz de consuion un poco mas clara
#  Devuelve todas las metricas "Precission" y "Recall" calculando la media de cada uno de ellos
from sklearn.metrics import classification_report

print( classification_report(y_test, clfK.predict(X_test) ) )

```


## Curva de ROC

Vemos los hay debajo de la curva de la grafica

```PYTHON

# Load the library
from sklearn.metrics import roc_curve,auc

# We chose the target
target_pos = 1 # Or 0 for the other class
fp,tp,_ = roc_curve(y_test,pred[:,target_pos])

# Dibujamos los resultados de la metrica
plt.plot(fp,tp)

# Vemos el dato devuelto por la metrica
auc(fp,tp)

```


## Precision VS Recall

ReCall: Es el número de elementos identificados correctamente como positivos del total de positivos verdaderos.
(Vemos los hay debajo de la curva de la grafica)

Precision: Es el número de elementos identificados correctamente como positivo de un total de elementos identificados como positivos.

Está claro que recall nos da información sobre el rendimiento de un clasificador con respecto a falsos negativos (cuántos fallaron), mientras que la precisión nos proporciona información sobre su rendimiento con respecto a los falsos positivos (cuántos capturados).

```PYTHON

# Imporatamos libreria para crear alas metricas
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

precision_score(y_test,clf.predict(X_test))

classification_report(y_test,clf.predict(X_test))

# Cross Validation
cross_val_score(clf,X,y,scoring="precision")
cross_val_score(clf,X,y,scoring="recall")

# Ejmplo de modelo SVC, entrenamiento y prediccion
wclf = SVC(kernel='linear', C= 1, class_weight={1: 10})
wclf.fit(X, y)
weighted_prediction = wclf.predict(X_test)

# Extraemos las metrris de ReCall y Precision
print 'Recall:', recall_score(y_test, weighted_prediction, average='weighted')
print 'Precision:', precision_score(y_test, weighted_prediction, average='weighted')

# Crear un informe de texto que muestre las principales métricas de clasificación
print '\n clasification report:\n', classification_report(y_test, weighted_prediction)

# Matriz de confusion
print '\n confussion matrix:\n',confusion_matrix(y_test, weighted_prediction)

# Accuracy
print 'Accuracy:', accuracy_score(y_test, weighted_prediction)

# El puntaje de F1 se puede interpretar como un promedio ponderado de la precisión y el recuerdo, 
# donde un puntaje de F1 alcanza su mejor valor en 1 y el peor puntaje en 0
print 'F1 score:', f1_score(y_test, weighted_prediction,average='weighted')

```
