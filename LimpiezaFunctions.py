# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:28:35 2021

@author: Adrián González
"""

#Librerías#

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from unicodedata import normalize
import plotly.graph_objects as go

#-----------------------------------------------------------------------------#

#Funciones de Normalización#

#Las funciones de normalización tienen por uso estandarizar los distintos 
#formatos usados en las distintas fuentes para llamar a los municipios.

def norma(s):
    
    trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
    s = normalize('NFKC', normalize('NFKD', s).translate(trans_tab))
    s = s.rstrip(' ')
    
    if s[0:3] == 'El ':
        
        s = s[3:]+', El'
        
    elif s[0:3] == 'La ':
        
        s = s[3:]+', La'
    
    elif s[0:4] == 'Los ':
        
        s = s[4:]+', Los'
        
    elif s[0:4] == 'Las ':
        
        s = s[4:]+', Las'
        
    elif s[-5:] == ' (LA)':
        
        s = s[:-5]+', La'
        
    elif s[-5:] == ' (EL)':
        
        s = s[:-5]+', El'
        
    elif s[-6:] == ' (LAS)':
        
        s = s[:-6]+', Las'
        
    elif s[-6:] == ' (LOS)':
        
        s = s[:-6]+', Los'
        
    s = s.upper()
        
    if s == 'VILLA DE OTURA':
        
        s = 'OTURA'
        
    if s == 'LANTEJUELA, LA':
        
        s = 'LANTEJUELA'
    
    return s

def norma2(s):
    
    if s == 'Villa de Otura':
        
        s = 'Otura'
    
    rep = (('á','a'),('é','e'),('í','i'),('ó','o'),('ú','u'),('ñ','n'),
           ('ä','a'),('ë','e'),('ï','i'),('ö','o'),('ü','u'))
    for a,b in rep:
                
        s = s.replace(a,b).replace(a.upper(),b.upper())
        
    res = ['La', 'El', 'Las','Los']
    
    for c in res:
        
        if c+' ' in s:
            
            s = s[len(c)+1:]+' '+c
            
    s = s.replace(' ','-').lower()
    
    return s

#-----------------------------------------------------------------------------#

#Funciones Scrapping#

#Las funciones de scrapping tienen como uso obtener mediante técnicas de web
#scrapping una serie de datos.

def Scrap1(A,B,id_tabla,url_0):
    
    """
    Función pensada para hacer webscrapping a la página web datoselecciones.com
    para las elecciones autonómicas por provincia. A es para el año electoral,
    B para la región, id_tabla el id de la tabla a conseguir, url_0 el comienzo 
    de la url de la página.
    """
    
    df = []
    
    for i in range(len(A)):
        
        url_1 = url_0+A[i]+'/'

        for j in range(len(B)):
            
            url = url_1+B[j]
                
            pagina = requests.get(url)
            soup = BeautifulSoup(pagina.content, 'html.parser')
            
            #Aquí se crea un except debido a que en el año 2004 la tabla de Cádiz
            #cambia de id
            
            try:
                
                tbl = soup.find('table', {'id': id_tabla[i]})
                df.append(pd.read_html(str(tbl))[0])
                
            except:
                
                tbl = soup.find('table', {'id': 'tb1_12'})
                df.append(pd.read_html(str(tbl))[0])
                    
    return df

def Scrap2(A,B,id_tab,url_0):
    
    """
    Función pensada para obtener por web scrapping los resultados de las elecciones
    generales por provincias, donde A es el vector con el año de cada elección,
    B es el vector con las provincias, id_tab el id de la tabla a conseguir, y
    url_0 el comienzo de la url objetivo.
    """
    
    df = []
    
    for i in range(len(A)):
        
        url_1 = url_0+A[i]+'/andalucia/'
        
        for j in range(len(B)):
            
            url = url_1+B[j]
            
            pagina = requests.get(url)
            soup = BeautifulSoup(pagina.content, 'html.parser')
            
            tbl = soup.find('table', {'id': id_tab[i]})
            df.append(pd.read_html(str(tbl))[0])
            
    return df

def Scrap_elpais(id_tabla):
    
    df = []
    
    id_pais = ['04.html','11.html','14.html','18.html','21.html','23.html',
                '29.html','41.html','']
    nueva_url = 'https://resultados.elpais.com/elecciones/2019-28A/generales/congreso/01/'
     
    for i in range(len(id_pais)):
                
        Url = nueva_url+id_pais[i]
            
        pagina = requests.get(Url)
        soup = BeautifulSoup(pagina.content, 'html.parser')
        
        tbl = soup.find('table', {'id': id_tabla})
        df.append(pd.read_html(str(tbl))[0])
        
    return df
    
def Scrap3(A, B, C, id_tab, url_0):
    
    """
    Función pensada para obtener vía web scrapping los resultados de las elecciones
    autonómicas en Andalucía por municipio, donde A es el vector con los años de
    las elecciones, B el vector con las provincias, C la lista de municipios, id_tab
    el id de la tabla y url_0 el inicio de la url de la página.
    """
    
    df = []
    
    for i in range(len(A)):
    
        url_1 = url_0+A[i]
        cont = 0
        
        for k in C:
            
            k = norma2(k)
            
            if cont <= 101:
                    
                url = url_1+B[0]+k
                    
            elif cont > 101 and cont <= 145:
                    
                url = url_1+B[1]+k
                    
            elif cont > 145 and cont <= 220:
                    
                url = url_1+B[2]+k
                    
            elif cont > 220 and cont <= 388:
                    
                url = url_1+B[3]+k
                    
            elif cont > 388 and cont <= 467:
                    
                url = url_1+B[4]+k
                    
            elif cont > 467 and cont <= 564:
                    
                url = url_1+B[5]+k
                    
            elif cont > 564 and cont <= 664:
                    
                url = url_1+B[6]+k
                    
            elif cont > 664:
                    
                url = url_1+B[7]+k
                
            pagina = requests.get(url)
            soup = BeautifulSoup(pagina.content, 'html.parser')
            
            try:
                
                tbl = soup.find('table', {'id': id_tab[i]})
                df.append(pd.read_html(str(tbl))[0])
                
            except:
                
                tbl = soup.find('table', {'id': 'tb1_12'})
                df.append(pd.read_html(str(tbl))[0])
                
            cont += 1
            
    return df

def Scrap4(cont, id_tabla, url_0):
    
    """
    Función pensada para obtener, vía web scrapping, los datos de renta por
    municipio de la AEAT, donde url_0 es la parte de la url igual para todos 
    los años, id_tabla es el id único para la tabla que pretende conseguirse,
    y cont es la parte de la url específica para cada año.
    """
    
    df = []
    
    for i in cont:
        
        url = url_0+i
        
        pagina = requests.get(url)
        soup = BeautifulSoup(pagina.content, 'html.parser')
        
        tbl = soup.find('table', {'id': id_tabla})
        df.append(pd.read_html(str(tbl))[0])

#-----------------------------------------------------------------------------#
                
#Funciones de Limpieza#

def ptoint(df):
    
    """
    Función para convertir los porcentajes de los resultados en números decimales
    """
    
    for i in range(len(df)):
        
        df[i]= df[i].stack().str.replace('%','').str.replace(',','.').unstack()
        df[i]['%'] = df[i]['%'].astype(float)
        df[i] = df[i][['Partidos','%']]
        
    return df
        
def Particip(df):
    
    """
    Función para calcular la participación a partir de las tablas de información
    general de las elecciones
    """
    
    Part = []
    
    for i in range(len(df)):
        
        df[i] = df[i].stack().str.replace('.','').unstack()
        
        if i == 13:
            
            df[i][1][[2,3]] = df[i][1][[2,3]].astype(int)
            Part.append(round(df[i][1][2]/(df[i][1][2]+df[i][1][3])*100,2))
            
        else:
            
            df[i][1][[3,4]] = df[i][1][[3,4]].astype(int)
            Part.append(round(df[i][1][3]/(df[i][1][3]+df[i][1][4])*100,2))
        
    return Part  

def Exclud(d,df):
    
    '''
    Función pensada para exluir de los dataframes de población aquellos municipios
    no incluídos en la lista de Municipios a escuchar. Las entradas son la lista
    de municipios y los dataframes de población. El return es la lista de dataframes
    sin los municipios sobrantes.
    '''
    
    for i in range(len(df)):
        
        for j in df[i]['Municipios'].unique().tolist():
            
            if norma(j) not in d:
                
                df[i] = df[i].drop(df[i][df[i]['Municipios'] == j].index)\
                    .reset_index(drop = True)
                    
    return df

#-----------------------------------------------------------------------------#

#Funciones de separación#

def Sep(df, Iz, Der):
    
    """
    Esta función sirve para unir los porcentajes de todos los partidos adscritos
    a cada bloque y separar ambos en dos vectores distintos.
    """

    Porcentaje_derecha = []
    Porcentaje_izquierda = []
    
    for i in range(len(df)):
        
        contador_derecha = 0
        contador_izquierda = 0
        
        for j in range(len(df[i])):
            if df[i]['Partidos'][j] in Der['Partidos'].tolist():
                
                contador_derecha += df[i]['%'][j]
                
            elif df[i]['Partidos'][j] in Iz['Partidos'].tolist():
                
                contador_izquierda += df[i]['%'][j]
                
        Porcentaje_derecha.append(round(contador_derecha,2))
        Porcentaje_izquierda.append(round(contador_izquierda,2))
        
    return Porcentaje_izquierda, Porcentaje_derecha

def Sep_prov(lim1, lim2, df1, df2):
    
    """
    Esta función sirve para crear un dataframe con los resultados por bloque 
    ideológico y provincia, diferencia entre ambos y ganador. Donde lim1 
    corresponde al primer parámetro en que se basará el dataframe, en este caso
    el año electoral, lim2 en el segundo, en este caso las provincias, y df1 y
    df2 son los vectores con los resultados de cada bloque.
    """
    
    I = np.zeros((len(lim1),len(lim2)))
    D = np.zeros((len(lim1),len(lim2)))
    
    Df1 = {}
    Df2 = {}
    
    for i in range(len(lim1)):
        for j in range(len(lim2)):
            
            I[i,j] = df1[j+9*i]
            D[i,j] = df2[j+9*i]
            
        Df1[lim1[i]] = [I[i].tolist()]
        Df2[lim1[i]] = [D[i].tolist()]
        Df1[lim1[i]] = Df1[lim1[i]][0]
        Df2[lim1[i]] = Df2[lim1[i]][0]
        
    Izq = pd.DataFrame(Df1)
    Izq.insert(0, 'Provincia', lim2)
    Der = pd.DataFrame(Df2)
    Der.insert(0,'Provincia',lim2)
    
    return Izq, Der

def Sep_Muni(df):
    
    '''
    Función pensada para separar los municipios en categorías en función de 
    la población que tenían en el año 1998. El valor de entrada es el dataframe
    con la población y el return devuelve un daataframe con una calificación 
    numérica de 1 a 5 según el tamaño de la población.
    '''
    
    a = []; b = []; c = []; d = []
    
    for i in range(len(df)):
        
        df[i]['Clasificación'] = 1
        
        a.append(df[i][(df[i]['Total']>=1000)&(df[i]['Total']<2000)&\
                       (df[i]['Periodo']==2005)]['Municipios'].unique().tolist())
        b.append(df[i][(df[i]['Total']>=2000)&(df[i]['Total']<10000)&\
                       (df[i]['Periodo']==2005)]['Municipios'].unique().tolist())
        c.append(df[i][(df[i]['Total']>=10000)&(df[i]['Total']<100000)&\
                       (df[i]['Periodo']==2005)]['Municipios'].unique().tolist())
        d.append(df[i][(df[i]['Total'] >= 100000) & (df[i]['Periodo'] == 2005)]\
                 ['Municipios'].unique().tolist())
            
        for j in df[i]['Municipios'].unique().tolist():
            
            if j in a[i]:
                
                df[i].loc[df[i]['Municipios'] == j,'Clasificación'] = 2
                
            elif j in b[i]:
                
                df[i].loc[df[i]['Municipios'] == j,'Clasificación'] = 3
                
            elif j in c[i]:
                
                df[i].loc[df[i]['Municipios'] == j,'Clasificación'] = 4
                
            elif j in d[i]:
                
                df[i].loc[df[i]['Municipios'] == j,'Clasificación'] = 5 
        
    return df
#-----------------------------------------------------------------------------#

#Funciones de visualización#

def fig1(x, y1, y2, y3):
    
    """
    Función pensada para dibujar un gráfico que incluya barras para representar
    los resultados de los bloques y lineas y puntos para representar la participación.
    donde el parámetro x corresponde al año electoral, y1 al resultado de la
    izquierda, y2 al de la derecha e y3 al de la participación.
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
        title = {'text': 'Resultados de las Elecciones al Parlamento de Andalucía',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title = 'Elección',
        yaxis_title = 'Porcentaje',
        legend_title = 'Resultado'
        )
    
    return fig

def fig2(Anio,Dif):
    
    """
    Función pensada para representar mediante un gráfico de barras la diferencia
    entre bloques ideológicos para elecciones, donde Anio es el año electoral
    y Dif es el vector con el valor de la diferencia.
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
            title = {'text': 'Diferencia de votos entre bloques ideológicos',
                     'y': 0.9,
                     'x': 0.5,
                     'xanchor': 'center',
                     'yanchor': 'top'},
            xaxis_title = 'Elección',
            yaxis_title = 'Porcentaje',
            legend_title = 'Ventaja de la'
            )

    return go.FigureWidget(data = bars, layout = layout)

def fig3(df):
    
    '''
    Funcioón pensada para graficar la evolución del peso de cada categoría de
    municipios en la población total.
    '''
    
    fig = go.Figure(data = [
        go.Bar(name = '< 1.000', x = df.loc['<1000'].index, y = df.loc['<1000'].values),
        go.Bar(name = '< 2.000', x = df.loc['<1000'].index, y = df.loc['<2000'].values),
        go.Bar(name = '< 10.000', x = df.loc['<1000'].index, y = df.loc['<10000'].values),
        go.Bar(name = '< 100.000', x = df.loc['<1000'].index, y = df.loc['<100000'].values),
        go.Bar(name = '>= 100.000', x = df.loc['<1000'].index, y = df.loc['>= 100000'].values)
    ])
    fig.update_layout(barmode = 'stack',
                      title_text = 'Evolución de la Población ',
                      xaxis_title = 'Año censal',
                      yaxis_title = 'Porcentaje',
                      legend_title = 'Habitantes')
    
    return fig

#-----------------------------------------------------------------------------#

#Funciones de agrupación#

def Agrup(df):
    
    '''
    Función pensada para reagrupar el dataframe de municipios por población en 
    otro por año y categoría. El valor de entrada es el dataframe de Población
    y el return el dataframe de porcentaje de población por año y categoría.
    '''
    
    P = []
    
    for i in range(len(df['Clasificación'].unique())):
        
        P.append(df[df['Clasificación'] == i+1][['Municipios','Periodo','Total']]\
                 .reset_index(drop = True))
        
    for i in range(len(P)):
        
        P[i] = P[i].groupby('Periodo').sum()
        
    keys = ['<1000','<2000','<10000','<100000','>= 100000']
    P = pd.concat(P, keys = keys).unstack()
    P = P['Total']
    
    for i in P.columns:
        
        Pob = []
        
        for j in range(len(P[i])):
            
            Pob.append(P[i][j]/P[i].sum()*100)
            
        for j in range(len(P[i])):
            
            P[i][j] = Pob[j]
            
    return P