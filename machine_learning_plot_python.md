# PLOT

## Funcion para pintar datos dado un modelo

Parametros
<li>modelo     :: modelo a pintar</li>
<li>X          :: DataFrame X</li>
<li>y          :: Vector y</li>
<li>separacion :: separacion de los piuntos del pintado</li>

```python

def draw(clf,X,y,h=0.05):
    
    plt.figure(figsize=(10,10))

    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + .05
    y_min, y_max = X[:, 1].min() - .05, X[:, 1].max() + .05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    if clf is not None:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.scatter(xx,yy,c=Z)
    else:
        plt.scatter(xx,yy)
    
    plt.scatter(X[:,0],X[:,1],c=y,cmap="Paired")

# llamada
draw(clf,X,y,h=0.01)

```


## Explicacion de un modelo usando SHAP

```python

import shap

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

shap.force_plot(explainer.expected_value[1], shap_values[1][6,:], feature_names=X.columns)

'''

## Plot de las dependeicas parciales

```python

# Importamos la libreria
from pdpbox import pdp

# Cargamos el plot con los datos
pdp_goals = pdp.pdp_isolate(model=clf, dataset=X_test, model_features=X.columns, feature='Glucose')

# titulo
pdp.pdp_plot(pdp_goals, 'Feature')

# Mostramos
plt.show()

``` 

## scatter

```python

plt.scatter(Vector_X, Vector_y)

```

# Histograma

```python

plt.hist(Vector)

´´´

## Ejemplo base

```python

# Ejemplo 1

plt.figure(figsize=(10,10))
plt.scatter(df["bumpiness"],df["grade"],c=df["target"])
plt.plot([0,0.7],[0.9,0])
plt.plot([0,0.75],[0.9,0])
plt.plot([0,0.7],[0.8,0])
plt.show()


# Ejemplo 2
df = pd.read_csv("data/terrain.csv")

X = df[["bumpiness","grade"]].values
y = df["target"]

plt.figure(figsize=(10,10))
plt.scatter(df["bumpiness"],df["grade"],c=df["target"])
plt.show()

```
