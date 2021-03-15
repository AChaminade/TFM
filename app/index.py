# -*- coding: utf-8 -*-

##### Librerías #####

import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output

##### Conectar con el archivo app #####

from app import app
#from app import server

##### Conectar con las páginas de la app #####

from apps import Intro, Analisis, Modelo

app.layout = dbc.Container([
    dcc.Location(id = 'url', refresh = False),
    dbc.Row([
        dcc.Link(
            'Introducción |',
            href = '/apps/Intro'
        ),
        dcc.Link(
            ' Análisis |',
            href = '/apps/Analisis'
        ),
        dcc.Link(
            ' Modelo', 
            href = '/apps/Modelo'
        )
    ]),
    dbc.Row(
        id = 'page-content',
        children = []
    )
], fluid = True)

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)

def display_page(pathname):
    
    if pathname == '/apps/Intro':
        return Intro.layout
    elif pathname == '/apps/Analisis':
        return Analisis.layout
    elif pathname == '/apps/Modelo':
        return Modelo.layout
    else:
        return Intro.layout
    
if __name__ == '__main__':
    app.run_server(debug = False)