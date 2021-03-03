# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:26:20 2021

@author: Adrián González
"""

##### Librerías #####

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from joblib import load
import pickle
import json

import pandas as pd
import numpy as np
import plotly.express as px

#-----------------------------------------------------------------------------#

##### Importación y preparación de los datos #####

df = pd.read_csv('Resultados/Data.csv')
df['Año'] = df['Año'].astype(str)
df['Año'][(df['Año'] == '2019') & (df['Mes'] == 11)] = '201911'

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
    
##### Creación de los mapas municipales #####

fig_gen = px.choropleth_mapbox(df[df['Tipo de Elección'] == 'General'],
                               locations = 'id',
                               geojson = municipio,
                               hover_name = 'Municipios',
                               animation_frame = 'Año',
                               color_continuous_scale = px.colors.diverging.balance,
                               color_continuous_midpoint = 0,
                               range_color = [-80, 80],
                               mapbox_style = 'carto-positron',
                               title = 'Evolución de las Elecciones Generales',
                               center = {'lat': 37.471325, 'lon': -4.581459},
                               zoom = 6.5, opacity = 0.5)

fig_aut = px.choropleth_mapbox(df[df['Tipo de Elección'] == 'Autonómica'],
                               locations = 'id',
                               geojson = municipio,
                               hover_name = 'Municipios',
                               animation_frame = 'Año',
                               color_continuous_scale = px.colors.diverging.balance,
                               color_continuous_midpoint = 0,
                               range_color = [-80, 80],
                               mapbox_style = 'carto-positron',
                               title = 'Evolución de las Elecciones Generales',
                               center = {'lat': 37.471325, 'lon': -4.581459},
                               zoom = 6.5, opacity = 0.5)

#-----------------------------------------------------------------------------#

##### Aplicación #####

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure = fig_gen
        )
    ])

if __name__ == '__main__':
    app.run_server(debug=True)