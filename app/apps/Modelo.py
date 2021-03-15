# -*- coding: utf-8 -*-

##### Librerías #####

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State

from app import app

import pandas as pd
import numpy as np
import pickle
from joblib import load

##### Modelo y Escalado entrenados, y otros datos #####

xgbo = pickle.load(open('../TFM/Resultados/final_model.pkl', 'rb'))
scaler = load('../TFM/Resultados/scaler.bin')
data = pd.read_csv('../TFM/Resultados/Data.csv')
municipios = data['Municipios'].unique()

##### aplicación #####

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Andalucía, ¿cambio estructural o caso aislado?',
                        className = 'text-center text-primary, mb-4'),
                width = 12
        )
    ]),
    dbc.Row([
        dbc.Col(html.H1('¿Que votará tu municipio?',
                        className = 'text-center text-primary, mb-4'),
                width = 12
        )
    ]),
    dbc.Row([
        dbc.Col(dcc.Markdown('''
                       Pero, después de todo, ¿puede predecirse qué votará un municipio
                       cualquiera en las próximas elecciones? 
                       
                       Rellenando las celdas que existen a continuación (las que tienen
                       asterisco son obligatorias), obtendrá una predicción para su 
                       municipio.
                       
                       Aunque el resto de las celdas pueden dejarse en blanco, esto 
                       implica que el modelo imputará los valores que considere, lo que 
                       disminuirá su eficacia.
                       
                       El modelo empleado, basado en la biblioteca de python XGBoost,
                       cuenta con un acierto del 94 % a la hora de predecir al bloque
                       ganador en un municipio, mientras que cuenta con un 82 % de acierto
                       a la hora de predecir el resultado exacto (Cálculo mediante 
                       la métrica de puntuación de varianza explicada).
                       
                       '''))
    ]),
    dbc.Row([
        dbc.Col(
            html.H2('Datos a nivel de Municipio',
                    className = 'text-center text-primary, mb-4'),
            width = 12
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id = 'Municipio',
                options=[{'label': city, 'value': city} for city in municipios],
                value = municipios[10]
            ),
        ),
        dbc.Col(
            dcc.Input(id = 'Población', type = 'number', placeholder = 'Población*',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'Paro_in', type = 'number', placeholder = 'Paro Inicial',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'Paro_fin', type = 'number', placeholder = 'Paro Final',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'renta', type = 'number', placeholder = 'Renta Disponible',
                      value = np.nan),
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Input(id = 'In_Af', type = 'number', placeholder = 'Inmigración Africana',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'In_Am', type = 'number', placeholder = 'Inmigración Americana',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'In_As', type = 'number', placeholder = 'Inmigración Asiática',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'In_Eu', type = 'number', placeholder = 'Inmigración Europea',
                      value = np.nan),
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.H2('Datos a nivel de Provincia',
                    className = 'text-center text-primary, mb-4'),
            width = 12
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Input(id = 'Del_Es', type = 'number', placeholder = 'Delincuencia Española',
                      value = np.nan)
        ),
        dbc.Col(
            dcc.Input(id = 'Del_Af', type = 'number', placeholder = 'Delincuencia Africana',
                      value = np.nan)
        ),
        dbc.Col(
            dcc.Input(id = 'Del_Am', type = 'number', placeholder = 'Delincuencia Americana',
                      value = np.nan)
        ),
        dbc.Col(
            dcc.Input(id = 'Del_As', type = 'number', placeholder = 'Delincuencia Asiática',
                      value = np.nan)
        ),
        dbc.Col(
            dcc.Input(id = 'Del_Eu', type = 'number', placeholder = 'Delincuencia UE',
                      value = np.nan)
        ),
        dbc.Col(
            dcc.Input(id = 'Del_Res', type = 'number', placeholder = 'Delincuencia No UE',
                      value = np.nan)
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.H2('Datos a nivel Comunidad',
                    className = 'text-center text-primary, mb-4'),
            width = 12
        )
    ]),    
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id = 'mes',
                multi = False,
                value = 1,
                options = [{'label': 'Enero', 'value': 1},
                           {'label': 'Febrero', 'value': 2},
                           {'label': 'Marzo', 'value': 3},
                           {'label': 'Abril', 'value': 4},
                           {'label': 'Mayo', 'value': 5},
                           {'label': 'Junio', 'value': 6},
                           {'label': 'Julio', 'value': 7},
                           {'label': 'Agosto', 'value': 8},
                           {'label': 'Septiembre', 'value': 9},
                           {'label': 'Octubre', 'value': 10},
                           {'label': 'Noviembre', 'value': 11},
                           {'label': 'Diciembre', 'value': 12}],
                placeholder = 'Mes*'
            ),
        ),
        dbc.Col(
            dcc.Dropdown(
                id = 'tipo',
                multi = False,
                value = 1,
                options = [{'label': 'Autonómicas', 'value': 0},
                           {'label': 'Generales', 'value': 1}],
                placeholder = 'Tipo de Elección*',
            ),
        ),
        dbc.Col(
            dcc.Input(id = 'ESO', type = 'number', placeholder = 'ESO',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'Bach', type = 'number', placeholder = 'Bachillerato',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'FP', type = 'number', placeholder = 'FP (media)',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Input(id = 'Sup', type = 'number', placeholder = 'Superior',
                      value = np.nan),
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id = 'gob',
                multi = False,
                value = 1,
                options = [{'label': 'Izquierda', 'value': 1},
                           {'label': 'Derecha', 'value': 2}],
                placeholder = 'Gobierno Autonómico*'
            ),
        ),
        dbc.Col(
            dcc.Dropdown(
                id = 'Event',
                multi = False,
                value = 2,
                options = [{'label': 'Ninguna', 'value': 2},
                           {'label': 'Golpe de Estado', 'value': 1},
                           {'label': 'Corrupción', 'value': 3},
                           {'label': 'Atentado', 'value': 4},
                           {'label': 'Repetición de elecciones', 'value': 5},
                           {'label': 'Moción de Censura', 'value': 6}],
                placeholder = 'Eventualidades'
            ),
        ),
        dbc.Col(
            dcc.Input(id = 'Temp', type = 'number', placeholder = 'Temperatura',
                      value = np.nan),
        ),
        dbc.Col(
            dcc.Dropdown(
                id = 'Prec',
                multi = False,
                value = 1,
                options = [{'label': 'No', 'value': 0},
                           {'label': 'Débil', 'value': 1},
                           {'label': 'Moderada', 'value': 2}],
                placeholder = 'Precipitaciones'
            ),
        )
    ]),
    dbc.Row([
        dbc.Col(
        html.Button('Enviar', id = 'button'),
        align = 'center'
        )
    ], justify = 'center'),
    dbc.Row(
        id = 'predict',
        children = 'Resultado de la predicción',
    ),
    dbc.Row([
        dcc.Markdown('''
            **Aclaraciones**
            
            Es necesario rellenar aquellos parámetros marcados con asterisco. Para el 
            resto no es necesario, ya que pueden ser imputados mediante el algoritmo,
            pero es probable que el rendimiento del modelo se resienta.
            
            Los datos de paro deben ser de tasa por población, no por población activa.
            
            La pandemia no se considera una eventualidad, al no haber habido efectos
            reseñables por la misma en las elecciones que han tenido lugar en ese
            espacio de tiempo.
            
            los datos de Paro, como los de inmigración, son sobre el total de la población,
            no sobre la población activa. La delincuencia se mide en número de condenados,
            no es un valor porcentual. La renta se mide en euros.
        ''')
    ])
])
                     
@app.callback(
    Output('predict', 'children'),
    [Input('button', 'n_clicks')],
    [State('Municipio', 'value'),
     State('Población', 'value'),
     State('Paro_in', 'value'),
     State('Paro_fin', 'value'),
     State('renta', 'value'),
     State('In_Af', 'value'),
     State('In_Am', 'value'),
     State('In_As', 'value'),
     State('In_Eu', 'value'),
     State('Del_Es', 'value'),
     State('Del_Af', 'value'),
     State('Del_Am', 'value'),
     State('Del_As', 'value'),
     State('Del_Eu', 'value'),
     State('Del_Res', 'value'),
     State('mes', 'value'),
     State('tipo', 'value'),
     State('ESO', 'value'),
     State('Bach', 'value'),
     State('FP', 'value'),
     State('Sup', 'value'),
     State('gob', 'value'),
     State('Event', 'value'),
     State('Temp', 'value'),
     State('Prec', 'value')]
)
def Prediction(n_clicks, Municipio, Pob, Paro_in, paro_fin, renta, In_Af, In_Am, In_As, 
               In_Eu, Del_Es, Del_Af, Del_Am, Del_As, Del_Eu, Del_Res, mes, tipo, ESO, 
               Bach, FP, sup, gob, Event, Temp, Prec):
    
    costa = data[data['Municipios'] == Municipio]['Costa'].unique()[0].astype(int)  
    muni = data['Municipios'].unique().tolist().index(Municipio)   
    Tipo = ['Autonómicas' if tipo == 0 else 'Generales']
    
    dt = [[18, mes, muni, Pob, tipo, renta, In_Af, In_Am, In_As, In_Eu, Del_Es, Del_Af,
          Del_Am, Del_As, Del_Eu, Del_Res, ESO, Bach, FP, sup, Paro_in, paro_fin, costa,
          gob, Event, Prec, Temp]]
    
    columns = ['Año', 'Mes', 'Municipios', 'Población', 'Tipo de Elección', 
               'Renta Disponible', 'Inmigración Africa', 'Imigración América', 
               'Inmigración Asia', 'Inmigración Europa', 'Delincuencia España', 
               'Delincuencia Africa', 'Delincuencia América', 'Delincuencia Asia', 
               'Delincuencia Unión Europea', 'Delincuencia Resto de Europa', 'ESO', 
               'Bachillerato', 'FP', 'Superior', 'Paro Total Inicial', 'Paro Total Final', 
               'Costa', 'Gobierno Autonómico', 'Eventualidades', 'Precipitaciones', 
               'Temperatura']
    
    X_test = pd.DataFrame(data = dt, columns = columns)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(data = X_test, columns = columns)
    prediction = xgbo.predict(X_test)[0]
    if prediction < 0:
        ganador = 'Derecha'
    else:
        ganador = 'Izquierda'
        
    if ganador == 'Izquierda':
        result = (100+prediction)/2
    else:
        result = (100-prediction)/2
    
    resultado =f'**En las próximas elecciones {Tipo[0]} en {Municipio} ganará el bloque de\
            la {ganador} con el {result:0.2f} % de los votos.**'
    
    return  resultado