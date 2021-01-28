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
        
    elif s[0:3] == 'EL ':
        
        s = s[3:]+', EL'
        
    elif s[0:3] == 'La ':
        
        s = s[3:]+', La'
        
    elif s[0:3] == 'LA ':
        
        s = s[3:]+', LA'
    
    elif s[0:4] == 'Los ':
        
        s = s[4:]+', Los'
        
    elif s[0:4] == 'LOS ':
        
        s = s[4:]+', LOS'
        
    elif s[0:4] == 'Las ':
        
        s = s[4:]+', Las'
        
    elif s[0:4] == 'LAS ':
        
        s = s[4:]+', LAS'
        
    elif s[-5:] == ' (LA)':
        
        s = s[:-5]+', La'
        
    elif s[-5:] == ' (EL)':
        
        s = s[:-5]+', El'
        
    elif s[-6:] == ' (LAS)':
        
        s = s[:-6]+', Las'
        
    elif s[-6:] == ' (LOS)':
        
        s = s[:-6]+', Los'
        
    if 'Ñ' in s:
        s = s.replace('Ñ','N')
        
    elif 'ñ' in s:
        s = s.replace('ñ','n')
        
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

def Correccion_Mun(df, d, dnorm):
    
    '''
    Función pensada para, dada una lista de municipios por provincia en un formato
    no deseado ordenados, cambiarlos por el formato correcto. Los valores de entrada
    son el dataframe o lista, la lista de municipios y la lista de municipios
    normalizada, y el return es el dataframe o lista modificado.
    '''
    
    for i in range(len(df)):
        for j in range(len(dnorm)):
            
            if norma(df['Municipios'][i]) == dnorm[j]:
                
                df.at[i, 'Municipios'] = d[j]
                
    return df

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
    el id de la tabla y url_0 el inicio de la url de la página. El return es una 
    lista de dataframes con los resultados y una lista de listas con datos de 
    """
    
    df = []
    Cuenta = []
    
    for i in range(len(A)):
    
        url_1 = url_0+A[i]+'/'
        cont = 0
        
        for k in C:
            
            k = norma2(k)
            
            if cont <= 101:
                    
                url = url_1+B[0]+'/'+k
                    
            elif cont > 101 and cont <= 145:
                    
                url = url_1+B[1]+'/'+k
                    
            elif cont > 145 and cont <= 220:
                    
                url = url_1+B[2]+'/'+k
                    
            elif cont > 220 and cont <= 388:
                    
                url = url_1+B[3]+'/'+k
                    
            elif cont > 388 and cont <= 467:
                    
                url = url_1+B[4]+'/'+k
                    
            elif cont > 467 and cont <= 564:
                    
                url = url_1+B[5]+'/'+k
                    
            elif cont > 564 and cont <= 664:
                    
                url = url_1+B[6]+'/'+k
                    
            elif cont > 664:
                    
                url = url_1+B[7]+'/'+k
                
            pagina = requests.get(url)
            soup = BeautifulSoup(pagina.content, 'html.parser')
            
            try:
                
                tbl = soup.find('table', {'id': id_tab[i]})
                df.append(pd.read_html(str(tbl))[0])
                Cuenta.append([A[i],C[cont]])
                
            except:
                
                tbl = soup.find('table', {'id': 'tb1_12'})
                df.append(pd.read_html(str(tbl))[0])
                Cuenta.append([A[i],C[cont]])
                
            cont += 1
            
    return df, Cuenta

def get_renta(cont, id_tabla, url_0):
    
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
        
    return df
        
def get_paro(Anio, num_Anio, mes, num_mes, prov, url_0):
    
    df = []
    
    for i in range(len(Anio)):
        for j in range(len(mes)):
            for k in range(len(prov)):
                
                try:
                
                    url = url_0+Anio[i]+'/'+mes[j]+'_'+Anio[i]+'/MUNI_'+prov[k]+\
                        '_'+num_mes[j]+num_Anio[i]+'.xls'
                    
                    df.append(pd.read_excel(url, skiprows = range(5)))
                    
                except:
                    
                    ar_0 = 'Data/Paro/MUNI_'
                    archivo = ar_0+prov[k]+'_'+num_mes[j]+num_Anio[i]+'.csv'
                    
                    df.append(pd.read_csv(archivo, sep = ';', skiprows = range(5),
                                          encoding = 'ISO-8859-1'))
                    
    return df

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

def Limp_Paro(df, d, prov):
    
    '''
    Función pensada para limpiar los datos de paro del SEPE. El valor de entrada 
    será la lista de dataframes df, la lista de municipios normalizados d y la lista
    de provincias, mientras que el return será ese mismo dataframe limpio.
    '''
    
    num_munip = [102, 44, 75, 168, 79, 97, 100, 105]
    
    for i in range(len(df)):
    
        # Primero se eliminarán las dos primeras filas que carecen de información.
        
        df[i] = df[i].drop(range(2)).reset_index(drop = True)
        
        #Ahora se escogerán las filas cuya columna de municipios no sea NaN
        
        df[i] = df[i][df[i]['Municipios'].notna()]
        
        # Se rellenarán los NaN con ceros
        
        df[i].fillna(0, inplace = True)
                
    #Ahora se eliminarán los municipios que no se encuentran en la lista
        
    df = Exclud(d,df)
    
    for i in range(len(df)):
        
        #Y a continuación se eliminará el exceso de filas de cada dataframe
        
        for j in range(len(prov)):
            
            if df[i]['Provincia'][0] == prov[j] and len(df[i])-num_munip[j] > 0:
                
                df[i].drop(df[i].tail(len(df[i])-num_munip[j]).index, inplace = True)
                
    return df

def Limp_Renta(df, d, dnorm, prov):
    
    '''
    Función pensada para limpiar los datos de Renta. El valor de entrada será la
    lista de dataframes df, la lista de municipios, la lista de municipios normalizados,
    y la lista de provincias. El return será esa misma lista con los datos limpios
    y columnas añadidas de la provincia a la que pertenece cada municipio y año
    al que pertenecen los datos.
    '''
    
    for i in range(len(df)):
        c = -1
        df[i]['Provincia'] = 'Almería'
        
        for j in range(len(df[i])):
            if norma(df[i]['Municipios'][j]) in dnorm:
                c += 1
            
            df[i]['Provincia'][j] = prov[c] #Asignar provincias a los municipios
            
        df[i]['Municipios'] = df[i]['Municipios'].str.strip('-0123456789 ')
            
        df[i] = df[i][df[i]['Puesto Nacional'].notna()].reset_index(drop = True) 
        #Eliminar las provincias
        
        if i == 5:
            
            df[i] = df[i][df[i]['Renta Bruta'] != 'S.E.'].reset_index(drop = True)
            #Notación única del último año
            df[i]['Renta Bruta'] = df[i]['Renta Bruta'].str.replace('.','').astype(int)
            df[i]['Renta Disponible'] = df[i]['Renta Disponible'].str.replace('.','')\
                                                                .astype(int)
            df[i]['Puesto Nacional'] = df[i]['Puesto Nacional'].str.replace('.','')\
                .str.replace('-','0').astype(int)       
            df[i]['Puesto Autonómico'] = df[i]['Puesto Autonómico'].str.replace('-','0')\
                .astype(int)      
            
            break                                       
        
        df[i]['Puesto Nacional'] = df[i]['Puesto Nacional'].str.replace('.','')\
                .str.replace('-','0').astype(int)
        df[i]['Puesto Autonómico'] = df[i]['Puesto Autonómico'].str.replace('-','0')\
                .astype(int)
        df[i]['Renta Bruta'] = (df[i]['Renta Bruta'])*1000
        df[i]['Renta Disponible'] = (df[i]['Renta Disponible']*1000).astype(int)
        
    return df
        
def strtoint(df,a,b):
    
    '''
    Función pensada para convertir a int todos los elementos de las columnas de 
    un dataframe seleccionado. El valor de entrada es el dataframe y la posición,
    de las columnas que se quiera modificar, mientras que el return es el mismo 
    dataframe modificado.
    '''
    
    for i in range(len(df)):
        for j in range(a,b):
            
            if type(df[df.columns[j]][i]) == str:
                if df[df.columns[j]][i] == ' ':
                    
                    df[df.columns[j]][i] = 0
                    
                else:
                    
                    df[df.columns[j]][i] = int(df[df.columns[j]][i].replace(',',''))
                    
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

def Sep_municipales(df, Iz, Der, elect):
    
    '''
    Función para calcular el porcentaje de cada bloque ideológico en las 
    elecciones fenerales a nivel municipal. Los valores de entrada son el dataframe
    con los resultados electorales, la lista con los partidos de izquierda y derecha
    y el vector con el año de elección, y el return el mismo dataframe con los 
    bloques por municipio.
    '''
    
    for i in range(len(df)):
        
        Izquierda = []
        Derecha = []
        
        for j in range(len(df[i].columns)):
            
            if df[i].columns[j] in Iz.tolist():
                
                Izquierda.append(df[i].columns[j])
                
            elif df[i].columns[j] in Der.tolist():
                
                Derecha.append(df[i].columns[j])
            
        df[i]['Resultado Izquierda'] = round(df[i][Izquierda].sum(axis = 1)/\
                                             df[i]['Votos válidos']*100,2)
        df[i]['Resultado Derecha'] = round(df[i][Derecha].sum(axis = 1)/\
                                           df[i]['Votos válidos']*100,2)
        df[i]['Diferencia'] = round(df[i]['Resultado Izquierda']-\
                                    df[i]['Resultado Derecha'],2)
        df[i]['Fecha'] = elect[i]

    return df
            

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

#Funciones de Cálculos varios#

def Calculo_porcentual(df1, df2, columnas):
    
    '''
    Función pensada para calcular el porcentaje de parados totales y por sector
    sobre la población total. Los valores de entrada son los dataframes con datos 
    de paro y de población, y las columnas que deben cambiarse. El return es el
    dataframe de Paro modificado.
    '''
    
    for i in range(len(df1)):
        
        a = int(df2[(df2['Año'] == df1['Año'][i]) & (df2['Municipios'] == \
                                                     df1['Municipios'][i])]['Total'])
        
        for j in columnas:
            
            df1[j][i] = df1[j][i]/a*100
            
    return df1
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

    return go.Figure(data = bars, layout = layout)

def fig3(df):
    
    '''
    Función pensada para graficar la evolución del peso de cada categoría de
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
                      xaxis_title = 'Año',
                      yaxis_title = 'Porcentaje',
                      legend_title = 'Habitantes')
    
    return fig

def fig4(df, n, k = 'Población'):
    
    '''
    Función pensada para realizar un gráfico de líneas con la población de las 
    provincias de Andalucía. Los valores de entrada son el dataframe con los datos,
    n un vector con las provincias, k el título del eje y. El 
    return es el gráfico en cuestión.
    '''
    
    fig = go.Figure()
    
    for i in n:
        
        fig.add_trace(go.Scatter(
            x = df.index,
            y = df['Total'][i],
            mode = 'lines',
            name = i,
        ))
        
    fig.update_layout(
        title = {'text': 'Evolución de la Población de las Provincias',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title = 'Año',
        yaxis_title = k,
        legend_title = 'Provincia'
        )
    
    return fig

def fig5(df):
    
    '''
    Esta función tiene la misma función, valga la redundancia, que la función
    anterior, pero para los municipios, destacando además a que provincia pertenecen.
    El valor de entrada es el dataframe con la población por municipios, y el return 
    es la figura.
    '''
    
    fig = go.Figure()
    
    for i in df['Municipios'].unique():
        
        fig.add_trace(go.Scatter(
            x = df['Periodo'].unique(),
            y = df[df['Municipios'] == i]['Total'],
            mode = 'lines',
            name = i,
            ))
        
    fig.update_layout(
        title = {'text': 'Población de los Municipios de más de 100 000 habitantes',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title = 'Año',
        yaxis_title = 'Población',
        legend_title = 'Municipios'
        )
    
    return fig

def fig6(df1, df2 = False, lab = 'Renta Disponible', modo = 'Clasificación', s = 5):
    
    '''
    Función diseñada para crear las figuras con las que explorar los datos de renta.
    Los valores de entrada son:
        
        -df1: Dataframe con los datos de renta.
        -df2: Dataframe con los datos de población. (Opcional)
        -lab: Rasgo que va a visualizarse, principalmente Renta Disponible y Puestos.
        -mode: el modo en el que se va a elegir los municipios a representar, 
               principalmente mediante la clasificación de municipios según población
               o según un top de puestos de los municipios.
        -s: En caso de que se decida realizar un top, el número de municipios, en caso
            de que se decida clasificación, el nivel.
        
    El return será la figura.
    '''
    
    fig = go.Figure()
    
    if modo == 'Clasificación':
        
        try:
            for i in df2[df2[modo] == s]['Municipios'].unique():
                Df = df1[df1['Municipios'] == i]
                    
                fig.add_trace(go.Scatter(
                    x = Df['Año'],
                    y = Df[lab],
                    mode = 'lines',
                    name = i,
                    ))
                
            fig.update_layout(
            title_text = 'Evolución de {}'.format(lab),
            xaxis_title = 'Año',
            yaxis_title = lab,
            legend_title = 'Municipios'
            )
                    
        except TypeError:
            print('No ha introducido los datos de población')
            return 'Falta de datos.'
        
    if modo == 'top':
        A = df1[(df1['Puesto Autonómico'] <= s) & (df1['Puesto Autonómico'] > 0)]
        for i in A['Municipios'].unique(): 
            #Elige s municipios
        
            fig.add_trace(go.Scatter(
                x = A[A['Municipios'] == i]['Año'],
                y = A[A['Municipios'] == i][lab],
                mode = 'lines+markers',
                name = i
                ))
            
        fig.update_layout(
            title_text = 'Evolución del top {}'.format(s),
            xaxis_title = 'Año',
            yaxis_title = lab,
            legend_title = 'Municipios'
            )
        
    if lab == 'Puesto Nacional' or lab == 'Puesto Autonómico':
        
        fig['layout']['yaxis']['autorange'] = 'reversed'
        
    return fig

def fig7(df, n, k = 'Andalucía'):
    
    '''
    Función pensada para realizar un gráfico de líneas y puntos de la población
    inmigrante. Los valores de entrada son el dataframe con los valores de inmigración,
    n un vector con las columnas a representar y k la región representada.
    '''
    
    fig = go.Figure()
    
    for i in n:
        
        fig.add_trace(go.Scatter(
            x = df.index,
            y = df['Total'][i],
            mode = 'lines+markers',
            name = i,
        ))
        
    fig.update_layout(
        title = {'text': 'Evolución de la Población Inmigrante de {}'.format(k),
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title = 'Año',
        yaxis_title = 'Población Inmigrante',
        )
    
    fig.update_xaxes(tickangle = 90)
    
    return fig

def fig8(df,n):
    
    '''
    Función pensada para graficar la evolución de la delincuencia por provincias.
    Los valores de entrada son el dataframe con los datos y un vector con las 
    provincias. El return es la figura.
    '''
    
    fig = go.Figure()
    
    for i in n:
        fig.add_trace(go.Bar(
            x = df['Año'].unique(),
            y = df[df['Provincia'] == i]['Total'],
            name = i
            ))
        
    fig.update_layout(
        barmode = 'stack',
        title = 'Evolución de la delincuencia {}'.format(df['Nacionalidad'].unique()),
        xaxis_title = 'Año',
        yaxis_title = 'Número de condenados'
        )
    
    return fig

def fig9(df):
    
    '''
    Función para representar los datos de educación. El valor de entrada es el 
    dataframe con los datos de educación. El return es la figura.
    '''
    
    fig = go.Figure()
    
    for i in df['Formación'].unique():
        
        fig.add_trace(go.Bar(
            x = df['Año'].unique(),
            y = df[df['Formación'] == i]['Total'],
            name = i
            ))
        
    fig.update_layout(
        barmode = 'stack',
        title = 'Evolución del Nivel de Estudios en Andalucía',
        xaxis_title = 'Año',
        yaxis_title = 'Porcentaje'
        )
    fig['layout']['xaxis']['autorange'] = 'reversed'
    
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