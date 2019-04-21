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

En este grafico podemos ver que hay correlacion positiva en los datos del 10 al 12, por lo que <b>5</b> nos puede valer de punto de partida en el paramretro <br>AR</b> del modelo.


## Creacion del modelo ARIMA

```python

# Importamos las lisbrerias basicas
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot

# Importamos la ibrerias para genarar el modelo ARIMA
from statsmodels.tsa.arima_model import ARIMA

# Funcion de chequeo de fecha durante la importacion del fichero
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# Leemos el fichero de entrada
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# fit model
# p : 5
# d : 1
# q : 0
# Creamos el modelo al que le indicamos la serie y los paramtros en base a los que debe de entrenar
model = ARIMA(series, order=(5,1,0))

# Entrenamos el modelo
model_fit = model.fit(disp=0)

# Visualizamos el resultado del modelo
print(model_fit.summary())

# Creamos un grafico con los errores residuales
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()

# Creamos aun grafico con la densidad de los valores de los errores
residuals.plot(kind='kde')
pyplot.show()

# Vemos la media, std, min, ...
print(residuals.describe())

# Vemos un resumen de los valores [ print(model_fit.summary()) ]
# 
#                              ARIMA Model Results
# ==============================================================================
# Dep. Variable:                D.Sales   No. Observations:                   35
# Model:                 ARIMA(5, 1, 0)   Log Likelihood                -196.170
# Method:                       css-mle   S.D. of innovations             64.241
# Date:                Mon, 12 Dec 2016   AIC                            406.340
# Time:                        11:09:13   BIC                            417.227
# Sample:                    02-01-1901   HQIC                           410.098
#                          - 12-01-1903
# =================================================================================
#                     coef    std err          z      P>|z|      [95.0% Conf. Int.]
# ---------------------------------------------------------------------------------
# const            12.0649      3.652      3.304      0.003         4.908    19.222
# ar.L1.D.Sales    -1.1082      0.183     -6.063      0.000        -1.466    -0.750
# ar.L2.D.Sales    -0.6203      0.282     -2.203      0.036        -1.172    -0.068
# ar.L3.D.Sales    -0.3606      0.295     -1.222      0.231        -0.939     0.218
# ar.L4.D.Sales    -0.1252      0.280     -0.447      0.658        -0.674     0.424
# ar.L5.D.Sales     0.1289      0.191      0.673      0.506        -0.246     0.504
#                                     Roots
# =============================================================================
#                  Real           Imaginary           Modulus         Frequency
# -----------------------------------------------------------------------------
# AR.1           -1.0617           -0.5064j            1.1763           -0.4292
# AR.2           -1.0617           +0.5064j            1.1763            0.4292
# AR.3            0.0816           -1.3804j            1.3828           -0.2406
# AR.4            0.0816           +1.3804j            1.3828            0.2406
# AR.5            2.9315           -0.0000j            2.9315           -0.0000
# -----------------------------------------------------------------------------

```

Vemos la grafica de los errores residuales, lo que nos indica que unapuede haber 
tendencias que no hayan sido capturadas por el modelo [ residuals = DataFrame(model_fit.resid) ]

<div align="center"><img src="imagenes/machin_learning_Series _Temporales_Plot3.png"/></div>


Vemos la desidad grafica de los errores residuales [ residuals.plot(kind='kde') ]

<div align="center"><img src="imagenes/machin_learning_Series _Temporales_Plot4.png"/></div>

En el resumen podemos ver un sesgo en la prediccion , una emedia NO cero en los residuos

```python
# count   35.000000
# mean    -5.495213
# std     68.132882
# min   -133.296597
# 25%    -42.477935
# 50%     -7.186584
# 75%     24.748357
# max    133.237980
```

## Crear pronosticos con ARIMA

Con ARMIA podemos generar pronosticos.

Para ello usaremos la fucnion <b>predict()</b> desde el objeto ARIMAResilts.

Vamos a dividor los datos en entrenamineto y test

podemos evitar todas estas especificaciones utilizando la función <b>forecast()</b>, que realiza un 
pronóstico de un solo paso utilizando el modelo.

La funcion forecast, es una función genérica para pronosticar a partir de series de tiempo o modelos de series de tiempo


```python

# Importamos las librerias basicas
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

# Importamos la libreria para crear ael modelo ARMINA
from statsmodels.tsa.arima_model import ARIMA

# importamos la libreria para usar el modelo forecast
from sklearn.metrics import mean_squared_error

# funcion de parseo de fechas para el fichero de entrada
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

#Leemos el fichero de entrada
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# Dividimos los datos en entrenamiento y test
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]

# Creaos una serie con los valores del entrenamoento
history = [x for x in train]

# Creamos una lista vacia para las almacenar las predicciones
predictions = list()

# recorremos los datos de TEST
for t in range(len(test)):

	# Creamos el modelo ARIMA
	#Con los valores de history, que guarda lo valores del entrenamiento
	model = ARIMA(history, order=(5,1,0))
	
	# Entrenamos el modelo
	model_fit = model.fit(disp=0)

	# Pasamos el modelo con el metodo forecast()
	output = model_fit.forecast()
	
	# Almaceno la prediccion de salida
	yhat = output[0]
	
	# Y almaceno la prediccion en la lista de predicciones
	predictions.append(yhat)
	
	# ME queda con la observacion actual
	obs = test[t]
	
	# Y almaceno/añado la observacion a la lista de valores con los que entreno el modelo
	history.append(obs)
	
	# Visualozo la prediccion y la ultima observacion
	print('predicted=%f, expected=%f' % (yhat, obs))

# Guardo la medio de los erroroes
error = mean_squared_error(test, predictions)

# Visualizo la media de los errores
print('Test MSE: %.3f' % error)

# Creo un grafico con los valores del TEST
pyplot.plot(test)

# Añado al grafico otra lina cn los valores e la prediccion
pyplot.plot(predictions, color='red')

# Muestro graficos en pantalla
pyplot.show()

```

Podemos tambine calcular el porcentaje el error cuadratico final (MSE) para las predicciones,
proporcionanado de este modo una comparacion con otras configuraciones de ARIMA

```python
# predicted=349.117688, expected=342.300000
# predicted=306.512968, expected=339.700000
# predicted=387.376422, expected=440.400000
# predicted=348.154111, expected=315.900000
# predicted=386.308808, expected=439.300000
# predicted=356.081996, expected=401.300000
# predicted=446.379501, expected=437.400000
# predicted=394.737286, expected=575.500000
# predicted=434.915566, expected=407.600000
# predicted=507.923407, expected=682.000000
# predicted=435.483082, expected=475.300000
# predicted=652.743772, expected=581.300000
# predicted=546.343485, expected=646.900000
# Test MSE: 6958.325
```

<div align="center"><img src="imagenes/machin_learning_Series _Temporales_Plot5.png"/></div>


```python

# A la hora de creara el modelo
model = ARIMA(history, order=(5,1,0))

# Podemos indicar aue calcule automaticamente los mejores valores
model = ARIMA(history, lambda = "auto")


```


## Ejemplo con codigo R

```R

eeadj <- seasadj(stl(elecequip, s.window="periodic"))
autoplot(eeadj) + xlab("Year") +
  ylab("Seasonally adjusted new orders index")

ggtsdisplay(diff(eeadj))

(fit <- Arima(eeadj, order=c(3,1,1)))

# otra opcion de arima es decirle que cualule el lambda autometicmanete
fit <- auto.arima(elecequip, lambda = "auto")

# chqueo de residuos
checkresiduals(fit)

# emviamos el resultado a un modelo forecast
fit %>% forecast %>% autoplot

```
