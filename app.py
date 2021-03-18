##### Librerías #####

import dash 
import dash_bootstrap_components as dbc

##### Crear app #####

app = dash.Dash(__name__,
                external_stylesheets = [dbc.themes.SOLAR],
                suppress_callback_exceptions = True,
                #meta_tags=[{'name': 'viewport',
                #            'content': 'width=device-width, initial-scale=1.0'}]
                )

#server = app.server