# Series Temporales

Los datos obtenidos de observaciones a lo largo del tiempo es una de los casos mas comunes.

Un aserie temporal es una secuencia de datos, observaciones o valores medidos en determinados momnetos y ordenador cronologicamnete.
Estos datos pueden estar especaidos en el tiempo en intervalos iguales o desiguales.

Una vez que tenemos los datos, se suelen analizar para identificar patrones en los datos y entender lo courrido a lo largo del tiempo.

Una de las acciones más comunes con este tipo de datos es la prediccion o o pronostico, como se suele hacer 
en bolsa, clima, demografías,...

## Caractereisticas

1) Son dependientes del tiempo.

2) Suelen tener una tendencia, es decir algun tipo de estacionalidad y/o variaciones propias de una periodo de tiemnpo determinado.

3) Suelen estar autocorrelacionadas.

## Series de tiempo estacionarias

Una serie temporal es "estacionaria" si sus propiedades no son afectadas por los cabios a lo largo del tiempo. Es decir:

1) La <b>media</b> de la serie no debe ser una funcion de tiempo, sino que debe de ser una constante.

2) La <b>varianza</b> tampoco depende del tiempo. (Numero de ondas en su grafico a lo largo del tiempo)

3) La <b>covarianza</b> tampo varia en funcion del tiempo. (Altura de sus ondas en su grafico)


<b>Importancia de las series estacionarias</b>

Intuitivamente partimos que de si una serie se comparta de una manera particular en el tiempo, lo seguirá haciendo igual en el futuro.

Como la mayoría de las series temporales en la vida real no son estacianarias y la probabilidada matematica esta mas desarrollada en este sentido, 
una de las tareas que se desarrollaron es en transformar una serie temporal en estaciaonaria o lo mas cercano posible.

## ARIMA

Uno de los modelos mas utilizados para realizar pronosticos sobre series temporales es ARIMA (AutoRegressive Integrated Moving Average)
Se abastece de una conjunto de de estructura de datos en series estandar y devulve un metodo simple para realizar pronosticos.

Es un sistema descriptivo y captura los aspectos clave del propio modelo.

<b>AR (Auto-Regresión)</b> :  Modelo que utiliza una relacion dependiente entre una serie de observaciones 
                      y el numero de observaciones anteriores.

<b>I (Integrado)</b>: Es uso de la diferenciacion de observaciones brutas, (por ejemplo, resta una observacion en el paso anterior)
               para hacer que la serie de tiempo sea <b>estacionaria</b>

<b>MA (media movil)</b> : Utiliza la dependencia entre observaciones y un error residual de una modelo de promedio movil aplicado a 
                  observaciones anteriores.

Cada uno de estos componenetes se especifica explicitamente en el modlo como parametro, para lo caul se utiliza 
la siguiente nomenglatura:

<b>p</b> : Numero de observaciones anteriores incluidas en el modelo, tamnbine llamada oprdes de retardo.

<b>d</b> : La cantidad de veces que se diferencian las observaciones brutas. Tambine llamado grado de diferenciacion

<b>q</b> : El tamaño de la ventana de promedio movil, tambine llamado orden de promedio movil.

Interiormente se crea una modelo de regresion lineal con los parametros p, d, q, y los datos se preparan medienta una 
grado de diferenciacion, lo que es lo mimso, se eleminan las estructuiras de tendencias y estacionalidad que puede 
afectar negativamente al modelo de regresion

En el siguiente ejemplo vamos acargar aun DataSet de ventas de una año.



```python

# Importamos librerias basicas
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

# funcion que nos parsea una fcha
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# Cargamos los datos 
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# Montramos las primera lineas de DataFrame
print(series.head())

# Veriamos algo así
# 
# 1901-01-01    266.0
# 1901-02-01    145.9
# 1901-03-01    183.1
# 1901-04-01    119.3
# 1901-05-01    180.3
# Name: Sales, dtype: float64

# Mostrariamos una grafica de los datos
series.plot()
pyplot.show()

````
<div align="center"><img src="imagenes/machin_learning_Series _Temporales_Plot1.png"/></div>

Podemos observar una tendencia clara de las ventas .
Esto nos indica qie la serie no estacionaria y para hacer que sea estacionaria necesitamos una deferenciaciacion.

Podemos tambine ver una grafiaco de autocorrelacion (esto tambine nos lo da PANDAS)

```python

# Carga de librerias basicas
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot

# libreria de chqueo y formato de las fechas del fichero de entrada
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# Lectura del fichero con los datos
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# Creamos y mostramos un grafico con los datos de autocorrelacion (funcion de PANDAS)
autocorrelation_plot(series)
pyplot.show()

```

<div align="center"><img src="imagenes/machin_learning_Series _Temporales_Plot2.png"/></div>

En este grafico podemos ver que hay correlacion en

