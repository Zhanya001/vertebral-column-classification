# Vertebral Column Classification

Este proyecto tiene como objetivo realizar la clasificación multiclase utilizando el conjunto de datos **"Vertebral Column Data Set"** del repositorio UCI Machine Learning. Se implementa un modelo de árbol de decisión, evaluando su rendimiento y comparando diferentes estrategias de poda.

## TABLA DE CONTENIDO

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Referencias](#referencias)
- [LinkedIn](#linkedin)

## DESCRIPCIÓN

El conjunto de datos "Vertebral Column" contiene características de columnas vertebrales y su clasificación en diferentes categorías. Este proyecto incluye los siguientes pasos:

1. **Carga de Datos**: 
   - Obtención del conjunto de datos utilizando la librería `ucimlrepo`.
2. **Preprocesamiento**:
   - División del conjunto de datos en conjuntos de entrenamiento y prueba.
   - Escalado de características con `StandardScaler`.
3. **Entrenamiento del Modelo**:
   - Entrenamiento de un modelo de árbol de decisión con `DecisionTreeClassifier`.
   - Optimización de hiperparámetros usando `GridSearchCV` y `RandomizedSearchCV`.
4. **Evaluación**: 
   - Medición del rendimiento del modelo mediante métricas de precisión, recall y un informe de clasificación.

## INSTALACIÓN

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install ucimlrepo pandas scikit-learn matplotlib
```
## USO

1. Clona el repositorio:

```bash
git clone https://github.com/Zhanya001/vertebral-column-classification.git
cd vertebral-column-classification
```

2. Ejecuta el script:
   
```bash
python  main.py
```

## CARGA DE DATOS

Se utiliza ```fetch_ucirepo``` para descargar el conjunto de datos, que se almacena en un DataFrame de Pandas.

## PREPROCESAMIENTO

Se dividen los datos en conjuntos de entrenamiento y prueba utilizando ```train_test_split```, asegurando que el 30% de los datos se reserve para la evaluación.

Se escalan las características con StandardScaler para mejorar la convergencia del modelo.

## ENTRENAMIENTO DEL MODELO

Se crea un modelo de árbol de decisión con ```DecisionTreeClassifier```.

Se utilizan dos enfoques para la optimización de hiperparámetros:

- GridSearchCV: Realiza una búsqueda exhaustiva sobre un conjunto definido de hiperparámetros.
- RandomizedSearchCV: Muestra un enfoque más eficiente al seleccionar aleatoriamente hiperparámetros de una distribución definida.

## EVALUACIÓN

Se evalúa el rendimiento del modelo final utilizando métricas como la precisión y se imprime un informe de clasificación.

## RESULTADOS

Los resultados obtenidos a partir de la búsqueda de hiperparámetros indican el mejor rendimiento del modelo. Los mejores hiperparámetros encontrados a través de ambos métodos se muestran en la salida del script, junto con el rendimiento en términos de precisión.


---------------------------------------------------------------------------

## REFERENCIAS

UCI Machine Learning Repository

LinkedIn
Conéctate conmigo en Ariet Michal





