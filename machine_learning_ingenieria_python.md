
## Dividir automaticamente de de los datos de un Dataframe

```python

# Importamos PANDAS
import pandas as pd
%pylab inline

# Leemos los datos del fichero de entrada
df = pd.read_csv("data/diabetes.csv")

# Seleccionamos las columnas que usareamos para predecir en un DataFrame
X=df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]

# Seleccionamos la columna a predecir en un vector
y=df['Outcome']

# Cargamos la libreria
from sklearn.model_selection import train_test_split

# Partimos los datos en train y test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

´´´


## Guadar/Cargar un modelo en/desde fichero

```python

# Cargamos la libreria
import pickle

# Guardamos el modelo en fichero
pickle.dump(clf,open("modelo.pkl","wb"))

# Recupero el modelo desde el fichero
clf_loaded = pickle.load(open("modelo.pkl","rb"))

```

## Reducir dimensionalidad de un vector

```python

df["SalePrice"]=np.log1p(df["SalePrice"])

```


# Limieza de Datos (IMPUTACION)

Vamos a tener valores con valor NANA y podemo usart tres sistemas para limpiar estos datos segun cada caso

## Interpolacion

```python

# Podemos rellenar los valores NAN con la media de los valores mas cercanos

# vemos los valores nulos
df.isnull

# me quedo con las distintos de 0, es dedir las que quiene valores nulos 
df.columns[ df.isnull().sum() != 0]

```


## Variables con valores Nan

```python

# Opcion 1: Dejar a None todos los valores NaN

for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
           "MasVnrType",
           'MSSubClass']:
    df[col]=df[col].fillna("None")



# Opcion 2: Dejar a None todos los valores 0

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars',
           'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
           "MasVnrArea"]:
    df[col]=df[col].fillna(0)
    
# Opcion 3: Podemos rellenar los valores NAN con la media de los valores mas cercanos
df.columns[df.isnull().sum()!=0]
df["LotFrontage"]=df.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))

# o podemos usar la funcion mode
df["Electrical"]=df["Electrical"].fillna(df["Electrical"].mode()[0])


```


# Variables escalares

```python

# Opcion 1

# one-hot encoding -> asigan valores algo del estilo para tres varaibles categoricas 100, 010, 001
# esto comando lo hace con todas las variable categoricas automaticamente
# y crea una nueva colimna con este nuevo valor y elimna las columnas originales de estas variables categoricas
# es la peor solucion!!!! solo es buena para variables categoricas que pueden ser solo 2 valores

dfb = pd.get_dummies(df)
dfb.head()

# Opcion 2

# otra forma de pasar las variables categoricas a numeros, es definir pesos
# reemplazando la categorias por mun numeros

# obtenemos los diferentes valores posibles de nuestra variable escalar
df["SaleCondition"].unique()

# asignamos la media de las veces que se repite la categoria en custion
df[ df["SaleCondition"] == 'Normal']["SalePrice"].mean()

# Opcion 3

# Sustituir cada variable por un vector con la libreria word2vec
# Esta opcion nos da la posibilidad de buscar vecionos 

```

## Pipelines

Esta libreria nos permite concatenar pasos

```python

# Cargamos la libreria 
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[("scaled",RobustScaler()),
                       ("rf",RandomForestRegressor(max_depth=4))
                       ]
               )

```
