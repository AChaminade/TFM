# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:22:06 2021

@author: Adrián González
"""

##Librerías##

import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#Clasificación#

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#Regresión#

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

#Boosting#

from xgboost import XGBRegressor
from xgboost import XGBClassifier

##LigthGBM##

import lightgbm as lgb

##Tensorflow##

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import tensorflow as tf

#-----------------------------------------------------------------------------#

##Clase##

class Modelo:
    
    '''
    Clase que sirve para preprocesar ligeramente los datos a emplear, principalmente
    a través de métodos de escalado, concretamente estandarización y normalización, y
    para la construcción de modelos de clasificación y regresión basados en los algoritmos
    de las bibliotecas Scikit-learn, Tensorflow, XGBoost y LightGBM. También permite
    evaluar dichos modelos a través de una serie de métricas provenientes de la librería
    Scikit-learn y, para algunos algoritmos concretos, permite la visualización, bien 
    del proceso, bien de la importancia de las características empleadas. Concretamente,
    los algoritmos que se emplearán en esta clase serán:
        
        Algoritmos de Clasificación:
        -----------------------------------------------------------------------
        (Todos ellos permiten visualizar matrices de confusión.)
        
        -> Regresión Logística: Sklearn (Visualización de características)
        -> SVC: Sklearn (No permite visualización de características)
        -> K-vecinos: Sklearn (No permite visualización de características)
        -> Bosques Aleatorios: Sklearn (Visualización de características)
        -> Compilación: Sklearn (No permite visualización de características)
        -> Redes Neuronales: Tensorflow (Visualización de función pérdida)
        -> Clasificador XGB: XGBoost (Visualización de características)
        -> Clasificador LightGBM: LightGBM (Visualización de características)
        -----------------------------------------------------------------------
        Algoritmos de Regresión:
        -----------------------------------------------------------------------
        
        -> Regresión lineal: Sklearn (Visualización de características)
        -> K-vecinos: Sklearn (No permite visualización de características)
        -> Regresor de Gradient Boosting: Sklearn (Visualización de características)
        -> Bosques Aleatorios: Sklearn (Visualización de características)
        -> Redes Neuronales: Tensorflow (Visualización de función de pérdida)
        -> Regresor XGB: XGBoost (Visualización de características)
        -> Regresor LightGBM: LightGBM (Visualización de características)
        -----------------------------------------------------------------------
        
    En cuanto a las métricas, podrán emplearse las siguientes:
        
        Para evaluar modelos de clasificación:
        -----------------------------------------------------------------------
        
        -> Matriz de confusión
        -> Reporte de Clasificación (que incluye las principales métricas para cada clase)
        -> Balance de la clasificación: Puntuación de 0 a 1 del modelo, definida como:
            
                      especifidad + sensibilidad
           balance =  --------------------------
                                  2
        -----------------------------------------------------------------------
        Para evaluar modelos de regresión:
        -----------------------------------------------------------------------
        
        -> Error absoluto medio
        -> Error cuadrático medio
        -> Varianza explicada: Puntuación de 0 a 1 definida como:
            
                       Var{y-y_pred}
            EVS = 1 -  -------------
                           Var{y}
        -----------------------------------------------------------------------
        
            Parámetros:
                df (pandas.DataFrame): Dataframe con los datos.
                tipo (str): Distinción entre clasificador y regresor.
                
            Return:
                Clase Modelo.
    '''
    
    def __init__(self, df, tipo = 'Clasificador'):
        
        self.__data = df
        self.tipo = tipo
        
        if tipo == 'Clasificador':
            
            self.y = self.__data['Ganador']
            self.X = self.__data.drop(['Ganador', 'Diferencia'], axis = 1) 
            
        else:
            
            self.y = self.__data['Diferencia']
            self.X = self.__data.drop(['Ganador', 'Diferencia'], axis = 1)
            
        self.__columns = self.X.columns
            
    ##### Preprocesado de datos #####
        
    def estandarizar(self):
        
        '''
        Función para estandarizar el conjunto de datos. Estandarización significa
        reescalar los datos para que tengan media de cero y desviación estándar de uno.
        '''
        
        self.Scaler = StandardScaler()
        self.X = self.Scaler.fit_transform(self.X)
        self.X = pd.DataFrame(data = self.X, columns = self.__columns)
            
    def normalizar(self):
        
        '''
        Función para normalizar el conjunto de datos. Normalización significa reescalar
        los datos para que se encuentren entre cero y uno.
        '''
        
        self.norma = MinMaxScaler()
        self.X = self.norma.fit_transform(self.X)
        self.X = pd.DataFrame(data = self.X, columns = self.__columns)
            
    def Split(self, size = 0.2):
        
        '''
        Función para separar los datos en conjuntos de entrenamiento y prueba
        '''
        
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size = size, random_state = 42)
            
    def retorno(self):
        
        '''
        Función para invertir el escalado de los datos de prueba.
        '''
        
        try:
            if self.Scaler:
                self.X_test = self.Scaler.inverse_transform(self.X_test)
                self.X_test = pd.DataFrame(data = self.X_test, columns = self.__columns)
                
        except NameError:
            if self.norma:
                self.X_test = self.norma.inverse_transform(self.X_test)
                self.X_test = pd.DataFrame(data = self.X_test, columns = self.__columns)
                
        except:
            print('No se han empleado métodos de reescalado.')
            
            
    ##### Modelos de Clasificación #####
    
    def NN_Clas_model(self, neuronas = [512,512,256,256,128], dropouts = [0.4,0.4,0.3,0.3],
                      epochs = 150, split = 0.2, size = 11640):
        
        '''
        Función para construir un modelo de clasificación basado en una red neuronal,
        mediante el módulo keras de tensorflow. Se establece, por resutados anteriores
        de prueba y error un número de 5 capas.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                neuronas (int): Lista con el número de neuronas por capa, excepto la
                    última, que solo tiene una.
                dropouts (int): Lista con la tasa de neuronas en drop-out por capa.
                epoch (int): Número de veces que la red recorre el dataset.
                split (int): Tasa de valores del conjunto de entrenamiento empleados como
                    validación.
                size (int): valor del batch_size.
                
            Return:
                self.model (modelo): Modelo de la red neuronal.
                self.history (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'NN'
        
        tf.keras.backend.clear_session()

        self.model = models.Sequential([
            
            layers.Dense(units = neuronas[0], input_dim = self.X_train.shape[1], 
                         activation = 'relu'), 
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[0]),
            
            layers.Dense(units = neuronas[1], activation = 'relu'),  
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[1]),
        
            layers.Dense(units = neuronas[2], activation = 'relu'), 
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[2]),
            
            layers.Dense(units = neuronas[3], activation = 'relu'), 
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[3]),
            
            layers.Dense(units = neuronas[4], activation = 'relu'),
            layers.LeakyReLU(),
            layers.Dense(units=1, activation = "sigmoid"),
            
        ],name="Modelo de Clasificación con Redes Neuronales",)
        
        self.model.compile(
            optimizer = optimizers.Adam(),
            loss = losses.binary_crossentropy,
            metrics = [metrics.binary_accuracy]
            )
        
        self.history = self.model.fit(self.X_train, self.y_train.tolist(),
                                      epochs = epochs, batch_size = size,
                                      validation_split = split, verbose = 1)
        
    def RFC(self, max_depth = 50, n_estimators = 150):
        
        '''
        Función para construir un modelo de clasificación basado en el algoritmo
        RandomForestClassifier.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                max_depth (int): Profundidad máxima de los árboles.
                n_estimators (int): Número de árboles.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'rfc'
        self.model = RandomForestClassifier(max_depth = max_depth, n_estimators =\
                                            n_estimators)
        self.model.fit(self.X_train, self.y_train)
        
    def SVClass(self, C = 16):
        
        '''
        Función para construir un modelo a partir de los algorimos de clasificación
        de maquinas de soporte de vectores.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                C (int): Parámetro de regularización.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'svc'
        self.model = SVC(C = C)
        self.model.fit(self.X_train, self.y_train)
        
    def LogReg(self, C = 5):
        
        '''
        Función para construir un modelo a partir del algoritmo de regresión logística
        de sklearn.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                C (int): Parámetro de regularización.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'logreg'
        self.model = LogisticRegression(C = C)
        self.model.fit(self.X_train, self.y_train)
        
    def KNN(self, n_neighbors = 5):
        
        '''
        Función para construir un modelo a partir del algoritmo de K-vecinos
        de sklearn.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                n_neighboors (int): Número de vecinos.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'knn'
        self.model = KNeighborsClassifier(n_neighbors = n_neighbors)
        self.model.fit(self.X_train, self.y_train)
        
    def StackModel(self):
        
        '''
        Función para construir un modelo a partir de los algoritmos SVClassifier,
        RandomForestClassifier y Regresión logística, mediante la función Stacking
        Classifier de sklearn.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'stack'
        estimators = estimators = [('svc', SVC()),
             ('rf', RandomForestClassifier(n_estimators=100, max_depth=50))]
        self.model = StackingClassifier(
            estimators = estimators, final_estimator = LogisticRegression()
            )
        self.model.fit(self.X_train, self.y_train)
        
    ##### Modelo de Regresión #####
    
    def NN_Reg_model(self, neuronas = [1024,512,512,256,256,128], epochs = 250, 
                      dropouts = [0.4,0.3,0.3,0.2,0.2], split = 0.2, size = 11640,
                      lr = 0.02, decay = 6e-4):
        
        '''
        Función para construir un modelo de clasificación basado en una red neuronal,
        mediante el módulo keras de tensorflow. Se establece, por resutados anteriores
        de prueba y error un número de 5 capas.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                neuronas (int): Lista con el número de neuronas por capa, excepto la
                    última, que solo tiene una.
                dropouts (float): Lista con la tasa de neuronas en drop-out por capa.
                epoch (int): Número de veces que la red recorre el dataset.
                split (float): Tasa de valores del conjunto de entrenamiento empleados como
                    validación.
                size (int): valor del batch_size.
                lr (float): Valor del learning rate.
                decay (float): Valor del decaimiento del ratio de aprendizaje.
                
            Return:
                self.model (modelo): Modelo de la red neuronal.
                self.history (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'NN'
        
        tf.keras.backend.clear_session()

        self.model = models.Sequential([
            
            layers.Dense(units = neuronas[0], input_dim = self.X_train.shape[1]), 
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[0]),
            
            layers.Dense(units = neuronas[1]),  
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[1]),
        
            layers.Dense(units = neuronas[2]), 
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[2]),
            
            layers.Dense(units = neuronas[3]), 
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[3]),
            
            layers.Dense(units = neuronas[4]), 
            layers.LeakyReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropouts[4]),
            
            layers.Dense(units = neuronas[5]),
            layers.LeakyReLU(),
            layers.Dense(units=1, activation = "linear"),
            
        ],name="Modelo de Regresión con Redes Neuronales",)
        
        self.model.compile(
            optimizer = optimizers.Adam(lr = lr, decay = decay),
            loss = losses.mae,
            metrics = [metrics.mse]
            )
        
        self.history = self.model.fit(self.X_train, self.y_train.tolist(),
                                      epochs = epochs, batch_size = size,
                                      validation_split = split, verbose = 1)
        
    def LinReg(self):
        
        '''
        Función para construir un modelo de regresión basado en regresión lineal.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'linreg'
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
    def GradBoost(self, n_estimators = 200, max_depth = 10, learning_rate = 0.3):
        
        '''
        Función para construir un modelo de regresión basado en Gradient Boosting.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
        
        self.model_type = 'gradboost'
        
        self.model = GradientBoostingRegressor(learning_rate = learning_rate,
                                               max_depth = max_depth,
                                               n_estimators = n_estimators)
        self.model.fit(self.X_train, self.y_train)
        
    def RFR(self, n_estimators = 150, max_depth = 20):
        
         '''
         Función para construir un modelo de regresión basado en Bosques aleatorios.
        
             Parámetros:
                 self.X_train (array): Array del conjunto de entrenamiento.
                 self.y_train (array): Array de la clasificación real.
                 n_estimators (int): Número de árboles.
                 max_depth (int): Profundidad máxima de los árboles
                
             Return:
                 self.model (modelo): Modelo entrenado.
         '''
        
         self.model_type = 'rfr'
    
         self.model = RandomForestRegressor(max_depth = max_depth,
                                            n_estimators = n_estimators)
         self.model.fit(self.X_train, self.y_train)
         
    def KNNR(self, n_neigbors = 12):
        
        '''
        Función para construir un modelo de regresión basado en K-vecinos.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                n_neigbors (int): Número de vecinos a tener en cuenta.
                
            Return:
                self.model (modelo): Modelo entrenado.
        '''
         
        self.model_type = 'knn'
         
        self.model = KNeighborsRegressor(n_neighbors = n_neigbors)
        self.model.fit(self.X_train, self.y_train)
        
    ##### LigthGBM #####
        
    def LGBModel(self, learning_rate = 0.5, max_depth = 15, n_estimators = 100,
                 epoch = 100):
        
        '''
        Función para construir un modelo basado en los algoritmos de LigthGBM, tanto
        clasificación como de regresión.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                learning_rate (int): Ratio de aprendizaje
                max_depth (int): Profundidad máxima de los árboles.
                n_estimators (int): Número de estimadores.
                epoch (int): Número de veces que se recorre el dataset.
                
            Return:
                self.model (modelo): Modelo de lgb.
        '''
        
        self.model_type = 'lgb'
        d_train = lgb.Dataset(self.X_train, label = self.y_train)
        
        params = {}
        params['learning_rate'] = learning_rate
        params['boosting_type'] = 'gbdt'
        params['max_depth'] = max_depth
        params['use_missing'] = False
        
        if self.tipo == 'Clasificador':
            params['objective'] = 'binary'
            params['metric'] ='binary_logloss'
            
        else:
            params['objective'] = 'regression'
            params['n_estimators'] = n_estimators
            
        self.model = lgb.train(params, d_train, epoch)
        
    ##### XGBoost #####
    
    def XGBmodel(self, learning_rate = 0.5, max_depth = 10, n_estimators = 100):
        
        '''
        Función para construir modelos de predicción basados en la librería XGBoost.
        
            Parámetros:
                self.X_train (array): Array del conjunto de entrenamiento.
                self.y_train (array): Array de la clasificación real.
                learning_rate (int): Ratio de aprendizaje
                max_depth (int): Profundidad máxima de los árboles.
                n_estimators (int): Número de árboles.
                
            Return:
                self.model (modelo): Modelo entrenado.
                
        '''
        
        self.model_type = 'XGB'
        
        if self.tipo == 'Clasificador':
            self.model = XGBClassifier(learning_rate = learning_rate, 
                                       max_depth = max_depth)
            self.model.fit(self.X_train, self.y_train)
            
        else:
            self.model = XGBRegressor(learning_rate = learning_rate,
                                      max_depth = max_depth,
                                      n_estimators = n_estimators)
            self.model.fit(self.X_train, self.y_train)
        
    ##### Predicción #####
    
    def pred_class(self):
        
        '''
        Función para la predicción de clases para los modelos de clasificación, y su 
        regularización respecto a los valores reales.
        
            Parámetros:
                self.X_test (array): Conjunto de prueba.
                self.model (modelo): Modelo entrenado.
                self.model_type (str): Tipo de modelo empleado.
                
            Return:
                self.y_pred (array): Predicciones del modelo.
        '''
        
        if self.model_type == 'NN':
            self.y_pred = self.model.predict_classes(self.X_test).reshape(-1)
            
        elif self.model_type == 'lgb':
            self.y_pred = self.model.predict(self.X_test).round(0)
            
        else:
            self.y_pred = self.model.predict(self.X_test)
            
    def pred(self):
        
        '''
        Función para la predicción en los modelos de regresión, y su regularización 
        respecto a los valores reales.
        
            Parámetros:
                self.X_test (array): Conjunto de prueba.
                self.model (modelo): Modelo entrenado.
                self.model_type (str): Tipo de modelo empleado.
                
            Return:
                self.y_pred (array): Predicciones del modelo.
        '''
        
        self.y_pred = self.model.predict(self.X_test)
        
    ##### Visualización #####
        
    def Graficar_Perdida(self):
    
        '''
        Función para graficar la función perdida del set de entrenamiento y el 
        de validación durante el proceso de entrenamiento de una red neuronal.
        
            Parámetros:
                self.model (model): Modelo entrenado.
                
            Return:
                fig (Figure): Gráfica.
        '''

        trace0 = go.Scatter(
                    y = self.history.history['loss'],
                    x = self.history.epoch,
                    mode = 'lines',
                    marker = dict(
                        color = "blue",
                        size = 5,
                        opacity = 0.5
                        ),
                    name = "Training Loss"
                )
    
    
        trace1 = go.Scatter(
                    y = self.history.history['val_loss'],
                    x = self.history.epoch,
                    mode = 'lines',
                    marker = dict(
                        color = "red",
                        size = 5,
                        opacity = 0.5
                        ),
                    name = "Validation Loss"
                )
    
        data=[trace0, trace1]
        
        fig = go.Figure(
                data=data,
                layout=go.Layout(
                    title = "Curva de aprendizaje",
                    yaxis = dict(title="Pérdida"),
                    xaxis = dict(title="Epoch"),
                    legend = dict(yanchor = 'top',
                                xanchor = 'center'
                    )
                )
            )
            
        return fig
    
    def Feature_importance(self, importance_type = 'gain', color = 'green'):
        
        '''
        Función para graficar la importancia de las características de un modelo.
        No vale para redes neuronales.
        
            Parámetros:
                self.model (model): Modelo entrenado.
                importance_type (str): Tipo de importancia a graficar.
                color (str): Color de representación de la gráfica.
                figsize (int): Tupla con las dimensiones deseadas de la figura.
                
            Return:
                fig (Figure): figura.
        '''
        
        if self.model_type == 'lgb':
            
            #Valores de las características del modelo lgb.
            valores = dict(zip(self.X_train.columns, 
                               self.model.feature_importance(importance_type = \
                                                             importance_type)))
            
        elif self.model_type == 'XGB':
            
            #Valores de las características del modelo de XGBoost.
            valores = self.model.get_booster().get_score(importance_type = importance_type)
            
        elif self.model_type == 'logreg':
            
            #Valores de las características del modelo de regresión logística
            valores = dict(zip(self.X_train.columns, self.model.coef_[0]))
            
        elif self.model_type == 'linreg':
            
            #Valores de las características del modelo de regresión lineal
            valores = dict(zip(self.X_train.columns, self.model.coef_))
            
        else:
            
            #Valores de las características de modelos basados en algoritmos de sklearn.
            valores = dict(zip(self.X_train.columns,
                               self.model.feature_importances_))
        
        #Ordenar los valores de mayor a menor.
        sorted_tuples = sorted(valores.items(), key=lambda item: item[1])
        valores = {k: v for k, v in sorted_tuples}
            
        fig = go.Figure(go.Bar(
            x = list(valores.values()),
            y = list(valores.keys()),
            orientation = 'h'
            ))
        fig.update_traces(
            marker_color = color,
            marker_line_color = 'black',
            marker_line_width = 1.5,
            opacity = 0.8
            )
        fig.update_layout(
            xaxis_title = 'Feature importance',
            yaxis_title = 'Feature',
            title = 'Importancia de las características',
            width = 900,
            height = 850
            )

        return fig
    
    def Plot_conf_mat(self, colorscale = 'Jet'):
        
        '''
        Función para graficar la matriz de confusión como un mapa de calor.
        
            Parámetros:
                self.y_test (array): Array con las observaciones reales.
                self.y_pred (array): Array con las predicciones.
                colorscale (str): Escala de color.
                
            Return:
                fig (Figure): Figura.
        '''
        
        z = confusion_matrix(self.y_test, self.y_pred)
        x = ['Izquierda', 'Derecha']
        y =  ['Izquierda', 'Derecha']

        z_text = [[str(y) for y in x] for x in z]
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, 
                                          colorscale = colorscale)

        fig.update_layout(title_text='<i><b>Matriz de confusión</b></i>',
                          xaxis_title = 'Predicción',
                          yaxis_title = 'Valor Real')

        fig.update_layout(margin=dict(t=50, l=200))
        fig['data'][0]['showscale'] = True
        
        return fig
    
    ##### Métricas #####
    
    def conf_mat(self):
        
        '''
        Función para calcular la especifidad y sensibilidad de un modelo de clasificación
        a partir de la función confusion_matrix de sklearn.
        
            Parámetros:
                self.y_test (array): Array con los resultados reales de la clasificación.
                self.y_pred (array): Array con los resultados del modelo.
                
            Return:
                conf_mat (array): Lista con los valores de la matriz de confusión, con
                    orden [TN, FP, FN, TN]
        '''
        
        return print(confusion_matrix(self.y_test, self.y_pred))
    
    def class_report(self):
        
        '''
        Función para recibir el reporte del modelo de clasificación.
        
            Parámetros:
                self.y_test (array): Array con los resultados reales de la clasificación.
                self.y_pred (array): Array con los resultados del modelo.
                
            Return:
                classification_report
        '''
        
        return print(classification_report(self.y_test, self.y_pred))
    
    def balance(self):
        
        '''
        Función para recibir el balance de precisión del modelo de clasificación.
        
            Parámetros:
                self.y_test (array): Array con los resultados reales de la clasificación.
                self.y_pred (array): Array con los resultados del modelo.
                
            Return:
                balanced_accuracy_score
        '''
        
        return print(balanced_accuracy_score(self.y_test, self.y_pred))
    
    def mae(self):
        
        '''
        Función para recibir el error absoluto medio del modelo de regresión.
        
            Parámetros:
                self.y_test (array): Array con los resultados reales de la clasificación.
                self.y_pred (array): Array con los resultados del modelo.
                
            Return:
                mean_absolute_error
        '''
        
        return print(mean_absolute_error(self.y_test, self.y_pred))
    
    def mse(self):
        
        '''
        Función para recibir el error cuadrático medio del modelo de regresión.
        
            Parámetros:
                self.y_test (array): Array con los resultados reales de la clasificación.
                self.y_pred (array): Array con los resultados del modelo.
                
            Return:
                mean_absolute_error
        '''
        
        return print(mean_squared_error(self.y_test, self.y_pred))
    
    def explain_variance(self):
        
        '''
        Función para recibir la explained variance score medio del modelo de regresión.
        
            Parámetros:
                self.y_test (array): Array con los resultados reales de la clasificación.
                self.y_pred (array): Array con los resultados del modelo.
                
            Return:
                explained_variance_score
        '''
        
        return print(explained_variance_score(self.y_test, self.y_pred))