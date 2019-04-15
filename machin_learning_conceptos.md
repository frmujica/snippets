# Regresion lineal

  Modelo utilizado para explicar la relacion entre una variable DEPENDIENTE Y y una serie de variables INDEPENTES X

```python

lm(data, varaiblea_a_predecir~varaible predictora)

Ejemplo

lm(data=data_frame, variable_a_predecir~Variable_predictora_1, Variable_predictora_2...Variable_predictora_N)

```


# Regresion logistica

Usada para clasificar.

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

  Es la media de dispersion mas comun, que nos indica como de dispersos estan los datos copn respecto a la media
  
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
   
