# Andalucía, ¿cambio estructural o caso aislado?

## 1. Objetivos del trabajo

Este trabajo tiene como objetivos recolectar y estudiar datos electorales, económicos y sociológicos de Andalucía para:
* Deducir si el cambio de Gobierno producido a partir de las elecciones de 2018 fue un caso aislado o la conclusión de un proceso más largo y profundo.
* Estudiar las causas que llevaron a lo anterior.
* Construir un modelo sólido que, además, coincida con las conclusiones del análisis.

## 2. Requerimientos

Para poder ejecutar el código de este repositorio será necesario tener instaladas las siguientes librerías:

**De Python**

* Pandas
* Numpy
* BeautifulSoup
* Requests
* Matplotlib
* Plotly
* Json
* Unicodedata
* Openpyxl (sólo si se tienen las versiones modernas de pandas)
* Math
* Impyute
* Scikit-learn
* XGBoost
* LightGBM
* Tensorflow
* Shap
* Joblib
* Pickle
* Dash
* dash_core_components
* dash_html_components
* dash_bootstrap_components

**De R** (usado sólo puntualmente para visualización, no necesario para reproducir el trabajo)

* Tmap
* Sp
* Readr
* Dplyr

## 3. Estructura del repositorio

El repositorio tiene una serie de carpetas, de las cuales las más importantes son:

* Data: Ahí se encuentran guardados todos aquellos datos que van a emplearse y no han sido obtenidos por scrapping.
* Resultados: Ahí se encuentran guardados los resultados de los notebooks, tanto los obtenidos de los primeros para emplearlos en los siguientes como los resultados finales, así como el modelo final.

El trabajo se divide en cuatro notebooks, cuya función es definida por su título. Puede verse el código directamente. Sin embargo, la biblioteca plotly, que ha sido usada de forma generalizada en la visualización de los datos, no puede ser representada directamente en GitHub, por lo que se recomienda ver los notebooks [1](https://nbviewer.jupyter.org/github/AChaminade/TFM/blob/master/1.%20Recolecci%C3%B3n%20y%20Limpieza%20de%20datos.ipynb), [3](https://nbviewer.jupyter.org/github/AChaminade/TFM/blob/master/3.%20An%C3%A1lisis%20de%20datos.ipynb) y [4](https://nbviewer.jupyter.org/github/AChaminade/TFM/blob/master/4.%20Modelado%20de%20datos.ipynb) en los dispuestos enlaces, para ser visualizados en nbviewer, que sí permite dichas gráficas.

Para el primer y último notebook se han creado dos módulos de apoyo, cuyo código puede verse en *LimpiezaFunctions.py* y *Modelar.py*.

El dashboard interactivo puede ejecutarse localmente ejecutando el código *index.py*, que dará un enlace a través del cual podrá verse el dashboard. Por motivos de simplicidad y comodidad, dado que el dashboard es pesado y, en consecuencia, lento a la hora de ejecutarse, se ha creado una aplicación web, que puede verse pinchando en el enlace https://electopred-andalucia.herokuapp.com/. Sin embargo, dado que heroku pone, para sus aplicaciones gratuitas, un límite de 500 MB, la aplicación del enlace es una versión recortada del dashboard original, que puede verse completo ejecutando localmente.
