# -*- coding: utf-8 -*-

##### Librerías #####

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

#-----------------------------------------------------------------------------#

##### Importación y preparación de los datos #####

df = pd.read_csv('Resultados/Data.csv')
df['Año'][(df['Año'] == 2019) & (df['Mes'] == 11)] = 201911

# Se escogen los resultados y el año

df = df[['Año', 'Diferencia', 'Municipios', 'Tipo de Elección']]

# Preparación de los datos para el mapa

d = pd.read_csv('Data/municipios_andalucia.txt')
d = d['Municipios']

# Datos geoespaciales de Carto
municipio = json.load(open('Data/ign_spanish_adm3_municipalities.geojson',
                           encoding = 'utf8'))

municipio_id = {}
for feature in municipio['features']:
    feature['id'] = feature['properties']['cartodb_id']
    for i in range(len(municipio)):
        municipio_id[feature['properties']['nameunit']] = feature['id']
mun_id = {}
for i in range(len(d)):
    try:
        mun_id[d[i]] = municipio_id[d[i]]
    except:
        pass
df['id'] = df['Municipios'].apply(lambda x: mun_id[x])

# Datos para las gráficas 
# Para todos los datos la primera lista corresponde a generales y la segunda a autonómicas
Iz = [[55.67, 58.9, 69.8, 70.33, 73.37, 67.9, 64.06, 58.1, 65.18, 60.62, 53.47, 56.45, 
       52.16, 50.71, 50.36],
      [68.32, 73.67, 75.87, 64.96, 65.65, 60.83, 66.35, 60.6, 58.56, 62.55, 48.6]]

De = [[44, 37.87, 30.05, 29.51, 26.62, 31.47, 35.85, 41.76, 34.6, 39.04, 46.19, 43.5,
       47.81, 48.29, 49.61],
      [31.6, 26.32, 24.14, 35.02, 34.35, 38.32, 33.54, 39.4, 41.22, 37.4, 51]]

Part = [[78.48, 68.65, 78.75, 70.77, 69.33, 76.2, 78, 68.77, 74.77, 72.77, 68.9, 69.08, 
         66.05, 73.31, 68.25],
        [66.31, 70.37, 55.34, 67.28, 77.94, 69.72, 74.66, 72.67, 62.23, 62.3, 56.56]]

Dif = [[round(i - j,2) for (i, j) in zip(Iz[0], De[0])],
       [round(i - j,2) for (i, j) in zip(Iz[1], De[1])]]

##### Parámetros #####

YEARS_1 = [1977,1979,1982,1986,1989,1993,1996,2000,2004,2008,2011,2015,2016,2019,201911]
YEARS_2 = [1982, 1986, 1990, 1994, 1996, 2004, 2008, 2012, 2015, 2018]

years_1 = [str(ints) for ints in YEARS_1]
years_2 = ['1982','1986','1990','1994','1996','2000','2004','2008','2012','2015','2018']

##### Funciones #####

def getmap(df):    
    
    '''
    Función para crear los mapas
    '''
    
    figure = dict(
        data = dict(
            locations = df.id,
            geojson = municipio,
            type = 'choroplethmapbox',
            z = df.Diferencia,
            colorscale = 'balance',
            text = df.Municipios,
            marker_opacity = 0.7,
            zmid = 0,
        ),
        layout = dict(
            mapbox = dict(
                layers = [],
                style = 'carto-positron',
                center = dict(
                    lat = 37.471325,
                    lon = -4.581459
                ),
                pitch = 0,
                zoom = 5.7,
            )
        )
    )
    fig = go.Figure(figure)
    return fig

def Voto(x, y1, y2, y3):
    
    """
    Función para dibujar un gráfico que represente con barras los resultados de 
    los bloques, y de lineas y puntos para representar la participación.
    """
    
    fig = go.Figure(data=[
        go.Bar(name='Derecha', x = x, y = y2, marker = {'color': 'blue'}),
        go.Bar(name='Izquierda', x = x, y = y1, marker = {'color': 'red'})
    ])


    fig.add_trace(
        go.Scatter(
            x = x,
            y = y3,
            name = 'Participación'
        )
    )

    fig.update_layout(
        xaxis_title = 'Elección',
        xaxis = dict(type = 'category'),
        yaxis_title = 'Porcentaje',
        legend_title = 'Resultado'
        )
    
    return fig

def Diferencia(Anio,Dif):
    
    """
    Función para representar mediante gráfico de barras la diferencia entre bloques.
    """
    
    bar_heigh = Dif
    labels = ['Ventaja de la Izquierda', 'Ventaja de la Derecha']

    color = {'Izquierda': 'red',
             'Derecha': 'blue'}

    df = pd.DataFrame({'y': bar_heigh,
                   'x': range(len(bar_heigh)),})
    df['labels'] = 0
    df['labels'][df['y']<0] = 'Derecha'
    df['labels'][df['y']>0] = 'Izquierda'
    df['y'][df['y']<0] = -df['y'][df['y']<0]

    bars = []

    for labels, label_df in df.groupby('labels'):
    
        bars.append(go.Bar(
            x = label_df.x,
            y = label_df.y,
            name = labels,
            marker = {'color': color[labels]}
            ))
    
    layout= dict(
            xaxis = dict(
                tickmode = 'array',
                tickvals = np.arange(len(Anio)).tolist(),
                ticktext = Anio
                ),
            xaxis_title = 'Elección',
            yaxis_title = 'Porcentaje',
            legend_title = 'Ventaja de la'
            )

    return go.Figure(data = bars, layout = layout)

#-----------------------------------------------------------------------------#

##### Aplicación #####

app = dash.Dash(__name__, external_stylesheets = [dbc.themes.SOLAR],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
    )

### Layout ###

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Andalucía, ¿cambio estructural  caso aislado?',
                        className = 'text-center text-primary, mb-4'),
                width = 12
        )
    ]),
    dbc.Row([
        dbc.Col(html.P('''
                       A lo largo de la historia de la democracia, Andalucía ha sido 
                       el bastión inexpugnable de la izquierda en España. Su poder allí 
                       ha sido tal que, a lo largo de 36 años, parecía imposible que el
                       Partido Socialista Obrero Español de Andalucía, su formación
                       hegemónica, fuera desalojada de la Junta. Hasta el año 2018, cuando
                       todo cambió. ¿Qué es lo que ha ocurrido para que se haya dado este
                       hecho, aparentemente inconcebible? ¿Se trata de un caso puntual,
                       o de la conclusión de un proceso más profundo?
                       '''))
    ]),
    dbc.Row([
        dbc.Col(html.H1('Evolución electoral de Andalucía',
                        className = 'text-center text-primary, mb-4'),
                width = 12
        ),
    ]),
    dbc.Row([
        dbc.Col([
            html.H5('Elecciones Generales',
                   className = 'text-center'),
            dcc.Graph(id = 'grafo1'),
            dcc.Slider(
                id = 'years-slider1',
                min = 0,
                max = len(YEARS_1)-1,
                value = 0,
                marks={year: str(YEARS_1[year]) for year in range(len(YEARS_1))},
            )
        ], width = {'size': 6}),
        dbc.Col([
            html.H5('Elecciones Autonómicas',
                   className = 'text-center'),
            dcc.Graph(id = 'grafo2'),
            dcc.Slider(
                id='years-slider2',
        		min=0,
        		max=len(YEARS_2)-1,
        		value=0,
        		marks={year: str(YEARS_2[year]) for year in range(len(YEARS_2))},
            )
        ], width = {'size': 6})
    ], no_gutters=True, justify='start'),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id = 'my-drop1',
                multi = False,
                value = 'Voto',
                options = [{'label': 'Voto por bloque + participación', 'value': 'Voto'},
                           {'label': 'Diferencia de voto entre bloques', 'value': 'Dif'}]
            ),
            dcc.Graph(id = 'grafo3')
        ], width = {'size': 6}),
        dbc.Col([
            dcc.Dropdown(
                id = 'my-drop2',
                multi = False,
                value = 'Voto',
                options = [{'label': 'Voto por bloque + participación', 'value': 'Voto'},
                           {'label': 'Diferencia de voto entre bloques', 'value': 'Dif'}]
            ),
            dcc.Graph(id = 'grafo4')
        ], width = {'size': 6})
    ]),
], fluid = True)


@app.callback(
    Output('grafo1', 'figure'),
    [Input('years-slider1', 'value')]
)
def update_map1(year):
    
    dff = df[df['Tipo de Elección'] == 'General']
    dff = dff[dff['Año'] == YEARS_1[year]]
    fig = getmap(dff)
    return fig

@app.callback(
    Output('grafo2', 'figure'),
    [Input('years-slider2', 'value')]
)
def update_map2(year):
    
    dff = df[df['Tipo de Elección'] == 'Autonómica']
    dff = dff[dff['Año'] == YEARS_2[year]]
    fig = getmap(dff)
    return fig

@app.callback(
    Output('grafo3', 'figure'),
    [Input('my-drop1', 'value')]
)
def update_figure1(tipo):
    
    if tipo == 'Voto':
        fig = Voto(years_1, Iz[0], De[0], Part[0])
    elif tipo == 'Dif':
        fig = Diferencia(years_1, Dif[0])
    return fig

@app.callback(
    Output('grafo4', 'figure'),
    [Input('my-drop2', 'value')]
)
def update_figure2(tipo):
    
    if tipo == 'Voto':
        fig = Voto(years_2, Iz[1], De[1], Part[1])
    elif tipo == 'Dif':
        fig = Diferencia(years_2, Dif[1])
    return fig

if __name__ == '__main__':
    app.run_server(debug = False)