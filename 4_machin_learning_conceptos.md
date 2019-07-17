# Machin Learning

## Aprendizaje supervisado

Proceso encargado de encontrar la relacion exitente entre unas variables de entrada y una variable de salida.
Es decir, decirle al algoritmo que respuesta deseamos frente a ciertos de datos de entrada, tras mostrarle muchos ejemplos.
Hasta el punto de que el algotimo puede dar un resultado correcto en funcion de unos datos de entrada sin haberlos visto anteriormente.


<table>

  <tr>
    <td><b>Regresion (metodos)</b></td>
    <td><b>Clasificacion (metodos)</b></td>
  </tr>
  
  <tr>
  <td>Regresion lineal</td>
  <td>Regresion Logistica</td>
  </tr>
  
  <tr>
  <td>Knn (vecinos)</td>
  <td>Knn (vecinos)</td>
  </tr>
  
  <tr>
  <td>Arboles</td>
  <td>Arboles</td>
  </tr>
  
  
  <tr>
  <td>Random-Forest</td>
  <td>Random-Forest</td>
  </tr>
  
  <tr>
  <td>Redes Neuronales</td>
  <td>Redes Neuronales</td>
  </tr>
  
  <tr>
  <td></td>
  <td>Navie Bayes</td>
  </tr>
  
</table>


## Aprendizaje NO supervisado (metodos)

Es el proceso que produce conocimineto desde los datos de entrada sin indicar el resultado que queremos obtener.
Lo que estamos buscando son comportamineto o patrones similares para generar conocimineto.

<table>
  <tr><td>Redes Neuronales</td></tr>
  <tr><td>Knn (vecinos)</td></tr>
  <tr><td>Clusterizacion</td></tr>
</table>

  

## Pasos de estudio de una problema con Machin Learning

1) Definicion del problema
2) Objetivos del Machin Learning
3) Recoleccion de los datos
4) Limpieza de datos
5) Entrenamineto y validacion de los modelos
6) Testeo
7) Publicacion


Datos de training : Son los datos con los que un modelo aprende
Datos de testeo : Son los datos con los que entreno el modelo
Datos e validacion : Son los datos con los que valido el modelo


## Definicion de metricas

- <b>Accurency</b> : Predicciones positivas en base al total de predicciones

  (Falsos_positivos + Verdaderos_Positivos) / (Verdaderos_positivos + Verdaderos_positivos + Falsos_Negativos + Falsos+positivos)

- <b>Precision</b> : Proporcion de predicciones positivas

  Verdaderos_positivos / (verdaderos_positivos + Falsos_Positivos)

- <b>Recall</b> : Total de aciertos

  Verdaderos_positivos / (Verdaderos_positivos + Falsos_Negativos)
  

# Regresion lineal

  Utilizaremos este modelo para predicir un dato que puede optar por muchos valores.
  
  Modelo utilizado para explicar la relacion entre una variable DEPENDIENTE Y y una serie de variables INDEPENTES X
  
```python

lm(data, varaiblea_a_predecir~varaible predictora)

Ejemplo

lm(data=data_frame, variable_a_predecir~Variable_predictora_1, Variable_predictora_2...Variable_predictora_N)

```


# Regresion logistica

Usada para predecir el resultado de una varaible categorica, es decir una varaible que tiene un numero limitado de valores posibles,
en funcion de las varaibles independientes o predictoras

En Python se define como funcion "sigmoind"

La definicion manual sería

```python

def sigmoid (array):
  res = 1/( 1 + np (-array) )
  
  return res

x = np.linespace(-10,10,100)
plt.plot(x, sigmoid (x))

```

La funcion automatica que podemos usar desde python

```python

gml(datos, Variable_a_predecir ~ lista_varialbes_predictoria)

# Ejemplo

glm(data=data_frame, 
    familiy="binomial",
    formukla=var_a_predecir~Var_predictora_1, Var_predictora_2....,Var_predictora_N,
    weightd=var1 + var2...)
```


# Gradiente descenciente

Usada para optimizar.

Es la elecccion del menor elemento en un conjunto de elementos

```Python

def derivada_theta(X,y):
  rerunt lambda theta_o, theta_1: sum((theta_0 + theta_1*X * y)/len(X)

```


# Desviacion Estandard

  Es la media de dispersion mas comun, que nos indica como de dispersos estan los datos con respecto a la media
  
  ```
      X     y     \
  a   1.2   5.5    |      En Python la funcion es:
  b   Nan   3.0    |>     DataFrame.std
  c   4.0   7.0    |
  d   Nan   Nan    |        X -> 3.95
  e   9.0   11,1   /        y -> 4,78
 
 ```
  
  
  # Mediana
  
    Son los valores centrales de nuestro vator de datos.
  
  ```
  1,3,7,9,14,20,22 = 9
  
  1,3,7,9,14,20 = 7 y 9 = (7+9)/2 = 3
  ```
  
  # Media
  
    Nos indica la tendencia de nuestros valores centrales
   
   ```
    X+X2+X3...Xn / tolla_numeros 
    ```
   
