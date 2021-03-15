# -*- coding: utf-8 -*-

##### Librerías #####

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output 

from app import app

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

##### Datos #####

data = pd.read_csv('../TFM/Resultados/Data.csv')
data['Ganador'] = np.where(data['Diferencia'] >= 0, 'Izquierda', 'Derecha')
data['Año'] = data['Año'].astype(str)
data['Año'][(data['Año'] == '2019')&(data['Mes'] == 11)] = '201911'

df1 = data[data['Tipo de Elección'] == 'General'][['Año','Municipios','Ganador']]\
                         .groupby(['Año', 'Ganador']).count().unstack()
df2 = data[data['Tipo de Elección'] == 'Autonómica'][['Año','Municipios','Ganador']]\
                         .groupby(['Año', 'Ganador']).count().unstack()
df3 = data[data['Tipo de Elección'] == 'General'][['Año','Población','Clasificación']].groupby(['Año','Clasificación'])\
        .sum().unstack()
df4 = df3.copy()
for i in df4.index:
    df4.loc[i] = df4.loc[i]/df4.loc[i].sum()*100
    
df = df = data.drop(['Año','Mes','Municipios','Provincia','Resultado Derecha','Resultado Izquierda','Tipo de Elección',
                'Clasificación','Puesto Nacional','Puesto Autonómico','Gobierno Nacional','Gobierno Autonómico','Analfabeto',
                'Primaria incompleta','Primaria','ESO','Bachillerato','FP','Superior','Partidos Izquierda','Partidos Derecha','Eventualidades',
                'Precipitaciones','Temperatura'], axis = 1)

df5 = df[['Diferencia','Población', 'Renta Disponible', 'Inmigración Africa', 'Imigración América', 'Inmigración Asia',
       'Inmigración Europa', 'Paro Agricultura Final', 'Paro Industria Final',
       'Paro Construcción Final', 'Paro Servicios Final','Ganador']]
Del = pd.read_csv('../TFM/Resultados/Delincuencia.csv')
Del = Del[Del['Nacionalidad'] == 'Total']
Provincia = ['Almería', 'Cádiz', 'Córdoba', 'Granada', 'Huelva', 'Jaén', 'Málaga', 'Sevilla']

##### Funciones de visualización #####

def figura1(df, tipo = 'Generales', prov = None):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = list(range(len(df))),
        y = df['Municipios']['Derecha'].values,
        mode = 'lines+markers',
        name = 'Derecha'
    ))
    
    fig.add_trace(go.Scatter(
        x = list(range(len(df))),
        y = df['Municipios']['Izquierda'].values,
        mode = 'lines+markers',
        name = 'Izquierda'
    ))
    
    if prov == None:
    
        fig.update_layout(
            title = {'text': 'Municipios por Ganador en Elecciones {}'.format(tipo),
                'y': 0.85,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title = 'Elección',
            yaxis_title = 'Número de Municipios',
            legend_title = 'Bloque'
        )
        
    else:
        
        fig.update_layout(
            title = {'text': 'Municipios por Ganador en Elecciones {} en {}'.format(tipo, prov),
                'y': 0.85,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title = 'Elección',
            yaxis_title = 'Número de Municipios',
            legend_title = 'Bloque'
        )
    
    fig.update_xaxes(tickangle = 45,
                     tickvals = list(range(len(df))),
                     ticktext = df['Municipios'].index)
    return fig

def figura2(df):
    
    fig = go.Figure()
    
    for i in np.sort(data['Clasificación'].unique()):
        fig.add_trace(go.Bar(
            x = list(range(len(df))),
            y = df['Población'][i].values,
            name = str(i)
        ))
        
    fig.update_layout(
        barmode = 'stack',
        title = {'text': 'Evolución de la población por grupo de municipio',
                'y': 0.85,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
        xaxis_title = 'Elección',
        yaxis_title = 'Población',
        legend_title = 'Grupo'
    )
    fig.update_xaxes(
        tickvals = list(range(len(df))),
        ticktext = df['Población'].index
    )
    
    return fig

def figura3(data):
    
    df = data[data['Tipo de Elección'] == 'General']
    df = df[df['Población'].notna()]
    Años = df['Año'].unique().tolist()
    Provincia = df['Provincia'].unique().tolist()
    
    ## Creación de la figura ##
    
    fig_dict = {
        'data': [],
        'layout': {},
        'frames': []
        }
    
    fig_dict["layout"]["xaxis"] = {"range": [1, 6], "title": 'Población', 'type': 'log'}
    fig_dict["layout"]["yaxis"] = {"title": "Diferencia de voto entre bloques", "type": "linear"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300, "easing": "quadratic-in-out"}}],
                    "label": "Inicio",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                    "label": "Pausa",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    
    # Creación de Sliders
    
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Año:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    # Creación de datos
    
    Año = '1977'
    for Prov in Provincia:
        data_por_año = df[df["Año"] == Año]
        data_por_año_prov = data_por_año[data_por_año["Provincia"] == Prov]
    
        data_dict = {
            "x": list(data_por_año_prov['Población']),
            "y": list(data_por_año_prov["Diferencia"]),
            "mode": "markers",
            "text": list(data_por_año_prov["Municipios"]),
            "marker": {
                "sizemode": "area",
                "sizeref": 200000,
                "size": 5
            },
            "name": Prov
        }
        fig_dict["data"].append(data_dict)
    
    # Creación de los frames para la animación
    
    for Año in Años:
        frame = {"data": [], "name": Año}
        for Prov in Provincia:
            data_por_año = df[df["Año"] == Año]
            data_por_año_prov = data_por_año[data_por_año["Provincia"] == Prov]
    
            data_dict = {
                "x": list(data_por_año_prov['Población']),
                "y": list(data_por_año_prov["Diferencia"]),
                "mode": "markers",
                "text": list(data_por_año_prov["Municipios"]),
                "marker": {
                    "sizemode": "area",
                    "sizeref": 200000,
                    "size": 5
                },
                "name": Prov
            }
            frame["data"].append(data_dict)
    
        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [Año],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
                "label": Año,
                "method": "animate"}
        sliders_dict["steps"].append(slider_step)
    
    fig_dict["layout"]["sliders"] = [sliders_dict]
    
    fig = go.Figure(fig_dict)
    fig.update_layout(yaxis_range=[-105,105],
                      title = {'text': 'Evolución del Voto en Municipios según Población',
                               'y': 0.85,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},)
    
    return fig

def figura4(df):
    
    fig = px.scatter_matrix(df, dimensions = df.drop(['Ganador'], axis = 1).columns, color = 'Ganador', width = 1650, height = 1300)
    fig.update_traces(diagonal_visible = False, showupperhalf = False)
    return fig

def figura5(df,n):
    
    fig = go.Figure()
    
    for i in n:
        fig.add_trace(go.Bar(
            x = df['Año'].unique(),
            y = df[df['Provincia'] == i]['Total'],
            name = i
            ))
        
    fig.update_layout(
        barmode = 'stack',
        title = 'Evolución de la delincuencia {}'.format(df['Nacionalidad'].unique()[0]),
        xaxis_title = 'Año',
        yaxis_title = 'Número de condenados'
        )
    
    return fig

##### Aplicación #####

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Andalucía, ¿cambio estructural  caso aislado?',
                        className = 'text-center text-primary, mb-4'),
                width = 12
        )
    ]),
    dbc.Row([
        dbc.Col(html.H1('Análisis de los datos',
                        className = 'text-center text-primary, mb-4'),
                width = 12
        ),
    ]),
    dbc.Row([
        dcc.Markdown(
            '''
            ¿Por qué la gente vota lo que vota? ¿Qué motivos se esconden detrás de las
            tendencias electorales? 
            
            A continuación, algunas respuestas.
            
            ## ¿Cómo se vota en los municipios?
            
            En primer lugar, para entender como ha evolucionado el electorado de Andalucía,
            ha que ver cómo han votado sus municipios. Concretamente, a lo largo de las 
            elecciones, en cuantos ha ganado cada bloque.
            '''
        )
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id = 'grafo2.1',
                     figure = figura1(df1) 
            ),
        ], width = {'size':6}),
        dbc.Col([
            dcc.Graph(id = 'grafo2.2',
                      figure = figura1(df2, 'Autonómicas') 
            ),
        ], width = {'size':6})
    ]),
    dbc.Row([
        dcc.Markdown(
            '''
            hay que tener en cuenta que, desafortunadamente, no se dispone de datos
            de todos los municipios en las elecciones autonómicas. 
            
            De todas formas, es obvio, con la excepción de las primeras elecciones generales,
            la izquierda ha ostentado un gran poder territorial en Andalucía. Esto está 
            relacionado con la dicotomía rural/urbano, que se da en Andalucía con más fuerza
            que en otras regiones de España.
            
            Es importante tener en cuenta que ha habido provincias (Cádiz, Córdoba,
            Jaén y, especialmente, Sevilla) en las que la izquierda ganó en todos los 
            municipios durante varias elecciones.
            
            Teniendo en cuenta que hasta en el periodo 2018/2019 la izquierda gana en la
            mayoría de los municipios, es necesario que la derecha reciba más votos en
            los municipios más poblados.
            '''
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
            id = 'grafo2.3',
            figure = figura3(data)
            )
        )
    ]),
    dbc.Row([
        dcc.Markdown(
            '''
            A partir de la gráfica se observa que, a partir del periodo 1989/1990, comienza
            a aumentar el voto a la derecha, especialmente en los municipios de más de 10k
            habitantes, hasta el punto de volverse mayoritarios en los municipios de ese tamaño.
            
            Por tanto, uno de los factores que puede decidir el Gobierno de la Junta es
            la distribución de la población.
            '''
        )
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id = 'grafo2.4',
                     figure = figura2(df3) 
            ),
        ], width = {'size':6}),
        dbc.Col([
            dcc.Graph(id = 'grafo2.5',
                      figure = figura2(df4) 
            ),
        ], width = {'size':6})
    ]),
    dbc.Row([
        dcc.Markdown(
            '''
            Donde las clasificaciones se definen como:
                
            1. Municipios menores de 1k habitantes.
            2. Municipios entre 1k y 2k habitantes.
            3. Municipios entre 2k y 10k habitantes.
            4. Municipios entre 10k y 100k habitantes.
            5. Municipios de más de 100k habitantes.
                
            Se observa un crecimiento de la población en Andalucía a lo largo de las 
            cuatro décadas de democracia, particularmente en los municipios de entre 10k y
            100k, y en menor medida en los de más de 100k, en detrimento de los municipios
            menores de 10k habitantes. 
            
            La evolución en la distribución de población parece beneficiar claramente a la
            derecha.
            
            **Correlación entre las características**
            '''
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'grafo2.6',
                figure = px.imshow(df.drop(['Ganador'], axis = 1).corr(), x = df.corr().columns, y = df.corr().columns, 
          color_continuous_scale = 'balance', color_continuous_midpoint = 0, width = 1500, height = 1000,
          title = 'Matriz de correlación')
            )
        )
    ]),
    dbc.Row([
        dcc.Markdown(
            '''
            Hay, en general, escasa correlación entre las variables, salvo para aquellas que
            están intrínsecamente relacionadas entre ellas, como la renta bruta y la disponible 
            o las distintas variables relacionadas con el paro.
            
            En general, las variables que más correlacionan linealmente con el sentido del
            voto:
                
            * La delincuencia de origen africano es la que más correlaciona, favoreciendo
                principalmente a la derecha política. 
            * La Renta correlaciona claramente en favor de la derecha.
            * El paro correlaciona favorablemente a la izquierda, de forma especial
                el Paro agrario, a excepción del Paro en el sector servicios, que 
                favorece a la derecha, puede que por ser el que más se da en los municipios
                grandes.
            * La población no parace tener, a priori, gran correlación con los 
                resultados.
                
            Lo anterior puede aclararse observando los resultados exactos por municipio.
            '''
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'grafo2.7',
                figure = figura4(df5)
            )
        )
    ]),
    dbc.Row([
        dcc.Markdown(
            '''
            A partir de aquí pueden extraerse o consolidarse las siguientes conclusiones:
            
            1. Existe un claro factor de clase en la distribución del voto.
            2. Los municipios con mayor tasa de paro tienden a votar más a la izquierda.
                * Aunque se habla mucho del impacto del P.E.R. y el paro Agrario en las
                victorias electorales del Partido Socialista, se ve cómo la izquierda
                se ve beneficiada por las tasas de paro en todos los sectores, de forma 
                similar, por lo que, de ser cierto el caso, no parece determinante.
            3. Existe una clara relación entre el voto a la derecha y la inmigración de todo 
            origen, aunque es menos evidente para el caso de la inmigración de origen
            americano, quizá por el idioma y la cultura.
            4. Parece claro que los municipios más pequeños son, también, los más frentistas
            en cuanto a posiciones políticas. Las diferencias en ellos tienden a ser mucho más
            extremas que en los grandes municipios urbanos. Es decir, hay una mayor, y más cerrada
            tradición de voto en los municipios pequeños, lo que tiende a favorecer a la 
            izquierda, que en los municipios grandes, donde existe más pluralismo y más
            oportunidad de que muten las mayorías, lo que favorece, últimamente, a la
            derecha.
            
            ## Pero, entonces, ¿Por qué ganó la derecha en 2018?
            
            Aunque no puede darse una respuesta tajante, debido al gran número de factores,
            pueden darse unas pautas que ayuden a entender, dado lo anterior, el hecho.
            
            Primeramente, debe tenerse en cuenta que las elecciones autonómicas de 2018
            coincidieron con la sentencia del caso E.R.E. de Andalucía, trama que salpicó
            especialmente al partido que había ostentado hasta entonces el Gobierno de la Junta,
            el PSOE-A, pero también a varios sindicatos y a tradicionales apoyos del PSOE-A,
            como Izquierda Unida.
            
            Aparte de ello, hay que tener en cuenta que, al darse por finalizada la crisis
            de 2008, la renta media llevaba creciendo 2014, especialmente en los grandes
            municipios.
            
            También la inmigración tocaba máximos en 2018-2019, a excepción de la de origen
            europeo.
            
            Pero, principalmente, hay que tener en cuenta lo siguiente:
            '''
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id = 'grafo2.8',
                figure = figura5(Del, Provincia)
            )
        )
    ]),
    dbc.Row([
        dcc.Markdown(
            '''
            La delincuencia tocó máximo en 2018, año de las elecciones, para todas las 
            nacionalidades. La bajada posterior se debe, sobre todo, a la bajada en la delincuencia
            de origen europeo comunitario, español y asiático. Para el resto de nacionalidades
            continuó creciendo en 2019.
            
            En definitiva, para las elecciones de 2018:
            
            * El Paro tocaba mínimos desde la crisis.
            * La Renta llevaba creciendo desde 2014.
            * Inmigración y delincuencia se encuentraban en máximos (a excepción de la de 
                                                                   origen europeo).
            * Los efectos de la corrupción que habían lastrado al bloque de la derecha en
            2015 se habían disuelto. En su lugar, ahora había ahora un gran atención mediática
            para un importante caso de corrupción que afectaba al bloque de la izquierda.
            * La distribución de la población favorecía más que en las anteriores un cambio
            de mayoría.
            * El hecho de que fuesen elecciones autonómicas, y que el caso E.R.E. fuese un
            caso autonómico penalizó al partido gobernante.
            
            Estos, junto a otros, fueron los principales factores que motivaron el 
            extraordinario resultado de la derecha política. Aunque, siendo una tendencia 
            ganadora desde hace tiempo, es razonable pensar que, en las circunstancias 
            actuales, la derecha podría consolidar su dominio.
            '''
        )
    ])
], fluid = True)

