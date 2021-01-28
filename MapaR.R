#Librerías y paquetes necesarios#

library(tmap)
library(sp)
library(readr)
library(dplyr)

#Mapa por provincias#

#Mapa y datos#

mapa <- readRDS("Data/gadm36_ESP_2_sp.rds")
data <- read_csv("Resultados/Resultado_provincias.csv")

#Añadir ganador por provincia#

data$Winner <- ifelse(data$Diferencia > 0, 'Izquierda','Derecha')

#Ajustar el nombre de las columnas del shapefile#

mapa = mapa[mapa$NAME_2 == c('Almería','Cádiz','Córdoba','Granada','Huelva','Jaén','Málaga','Sevilla'),]
names(mapa)[7] <- 'Provincia'

#Unir datos y mapa#

mapadata <- merge(mapa,data, by = 'Provincia', duplicateGeoms = TRUE)

#Creación del mapa animado#

mp <- tm_shape(mapadata[mapadata$Winner == 'Derecha',]) +
  tm_polygons(col='Derecha', title = "Derecha", palette = "Blues", style = 'cont',
              breaks = c(30,40,50,60,70,80,100))+
  tm_shape(mapadata[mapadata$Winner == 'Izquierda',]) +
  tm_polygons(col='Izquierda', title = "Izquierda", palette = "Reds", style = 'cont',
              breaks = c(30,40,50,60,70,80,100))+
  tm_facets(along = "Elección", as.layers = TRUE)+
  tm_layout(legend.outside = TRUE)+
  tm_borders()

animation_tmap(mp, filename="Mapas/Mapa_provincias.gif", width = 1600, height = 1300, 
               dpi = 350, delay=100)
