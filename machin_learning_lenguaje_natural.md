# Interpretacion del lenguaje natural (NLP)

## Problema de clasificacion

Es una disciplina que une Machin lerning, Bigdata, Data Science que intenta procesar e interpretar texto.

Disponemos de varios niveles de analisis


## Nivel de tokerizacion 

  Si tenemos una documento la partimos en frases y si tenemos una frase la partimos en tokens
despues evaluamos los signos de puntuacion, para poder dar sentido a una frase o conjunto de tokens
  Analizamos las negaciones, por ejemplo en ingles cant's puede ser un tocken o dos


## Nivel de Analisis morfologico
 
  En ese nivel, determinamos que tipo de palabra es, nombre, advervio, verbos, artuculo...
 
 Esto nos puede indicar si el texto es positivo o negativo, asuminedo que los objetos son nombres y dando peso a los adjetivos
 
 Este parte tambien detecta el tiempo verbal y la forma
 
 Tambine analiza el contexto ya que hay palabrar que puedne significar dos cosas diferentes en base al contexto
 
 
## Nivel de Stemming and lemmatizacion

  Stemming: es obtener la raiz de la palabra, para poder obtener el significado para agrupar significados de 
este modo se reduce el vocabulario y por tanto el trabajo. 
  Este sistema elimina informacion como el tiempo verbal.
  
  lemmatizacion: es lo mismo que Stemming, pero en vez de cortar y quedarse con el principio, este sistema asigna correcta
para agrupar las palabras. (cenar = Cenaremos, cenariamos, cenemos)
 

## Nivel NER o Reconocimimneto de entidades NOMBRADAS

  Reconocimineto de ciudades, paises, fechas, años, datos que se reconocen como nombres, sistema para detectar personas, empresas, fechas...
  Busca y reconoce entidades

  Estos sistemas manualmente, o con expresiones regulares o etiquetamos datos mediante una clasificador
 
 
## Nivel Analisis sintactico

  Detectar el sujeto, precicado, complemento directo... Esto nos da la categorio y la funcion

  categoria: Nos dice cual es la palabra de referencia o de mas pero en un grupo de tokens
  
  referencia: que es la palabra, sujeto, predicado, complemento...

  Nos permite conocer como se afectan los tokens entre si


## Nivel Analisis semantico

  Entender el significado de la frase


## Nivel Pragmatics and Inference

  Pragmatics: Busqueda de referencias en la frase que puede que no estan en la frase.
  Busca referencia que esten o no en la frase.

  Inference -> implicaciones o acciones que se dan por conocidas aunque no se digan en la frase.
  valores o informacion que se dan como entendidas aunque no estén expresadas en la frase.





El texto que pasemos al modelo debe ter algo de estructura por lo que debe haber un preprocesado de los datos

Para ello usaremos herraminetas:

* Sistema Bag of words (es es una modelo de analisis)

    Tenemos un texto en donde cada palabra es una vector 
    Agruparemos las palabras en un vector, con palabras unicas y el numero de veces que se repite para reducir el numero d epalabras


* Stop Words

    Eliminamos palabras que no tiene contenido, como articulos, pronombres...

* Agrupamos que son sinonimoas


* TF-IDF o frecuencia de repeticion

    En este sistema vemos en cuentos documnetos aparece un termino y lo dividimos entre el numero de total de documentos
    
    Si tenemos un vector en donde cada posicion del vector tenemos una palabra
    
    Asiganaremos una peso a ciertas palabras.


* Nibgrams and Trigrams

    Agrupa palabras para buscar sentidos a las frases     


* Word Embedding

     Creariamos una vector numerico por cada palabra, que nos permitiria que palabras parecidas estén cerca  
     lo que nos permitirá buscar vecinos cercanos
    


# Algoritmos de clasificacion utilizadas

      regresion logistica
      random forest
      SVM
      Deep Neurunal
      Networks


# Metricas mas utilizadas: 

    <b>Precission</b>              : De los que digo que si, cuantos son que si
    <b>Recall</b>                  : De los posivicos cunatos son positivos 
    <b>Accuracy</b>                : Cuantos acierto en global, si esta bien valanceados los datos
    <b>AUC(Curva ROC)</b>          : Area bajo la curva ROC, cunato me equivo en funcion de cunato acierto
    <b>Precission-Recall curve</b> : Mezcla de Precission y Recall


https://likegeeks.com/es/tutorial-de-nlp-con-python-nltk/



```python

# Carga de datos

# Let's do 2-way positive/negative classification instead of 5-way    
def load_sst_data(path,
                  easy_label_map={0:0, 1:0, 2:None, 3:1, 4:1}):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            example = {}
            example['label'] = easy_label_map[int(line[1])]
            if example['label'] is None:
                continue
            
            # Strip out the parse information and the phrase labels  we don't need those here
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)
            
            
            # print one line of the file:
            if i==1:
              print('Example line:')
              print(len(line))
              print(line)
              print(example['text'])
              
    data = pd.DataFrame(data)
    return data

  
training_set = load_sst_data(sst_home + 'train.txt')
dev_set = load_sst_data(sst_home + 'dev.txt')
test_set = load_sst_data(sst_home + 'test.txt')

print('Training size: {}'.format(len(training_set)))
print('Dev size: {}'.format(len(dev_set)))
print('Test size: {}'.format(len(test_set)))

# Exploracion de los datos
training_set[training_set.label == 0].head(10)

    label       text
22  0           This is n't a new idea .
34  0           ... a sour little movie at its core ; an explo...
37  0           Made me unintentionally famous    as the queas...
52  0           The modern-day royals have nothing on these gu...
53  0           It 's only in fairy tales that princesses that...
59  0           An absurdist spider web .
76  0           By no means a slam-dunk and sure to ultimately...
110 0           It 's not a great monster movie .
144 0           Too often , Son of the Bride becomes an exerci...
148 0           A party-hearty teen flick that scalds like acid .


# Exploracion de valoraciones positivas
training_set[training_set.label == 1].head(10)

  label       text
0 1           The Rock is destined to be the 21st Century s...
1 1           The gorgeously elaborate continuation of Th...
2 1           Singer composer Bryan Adams contributes a sle...
3 1           Yet the act is still charming here .
4 1           Whether or not you re enlightened by any of D...
5 1           Just the labour involved in creating the layer...
6 1           Part of the charm of Satin Rouge is that it av...
7 1           a screenplay more ingeniously constructed than...
8 1           Extreme Ops  exceeds expectations 
9 1           Good fun , good action , good acting , good di...

# Vemos el valanceo de los datos 
training_set.groupby('label').size().reset_index(name='n')

  label       n
0 0           3310
1 1           3610

```

De nuestos datos de entrenamiento, vemos el balanceo de los datos

Vamos a crear un primer modelo de una DataSet que tenemos en una libreria llamada nltk

```python

import nltk
nltk.download("popular")

from nltk import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_auc_score, precision_recall_curve, roc_curve

```

https://pythonspot.com/category/nltk/

https://likegeeks.com/es/tutorial-de-nlp-con-python-nltk/







