# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:18:46 2023

@author: User
"""
import streamlit as st
import joblib
import numpy as np 
from PIL import Image
import sklearn
import pandas as pd
#=========================================================================
#           DETALLES DE LA PAG WEB 
#=========================================================================
st.set_page_config(page_icon='🏘️',page_title='Predicción de Precios Inmobiliarios')
#=========================================================================
#           CARGAR Y LEER EL MODELO
#=========================================================================

#   Cargamos el modelo desde el archivo
model_filename = 'house_prices.pkl'
loaded_model = joblib.load(model_filename)
print("Hemos Cargado el modelo...")

#=========================================================================
#=========================================================================
# Crear una barra lateral
#sidebar = st.sidebar

# Añadir un título a la barra lateral
#sidebar.title("Modelo de estimación de precio de Casas con ML")

# Añadir un botón a la barra lateral
#if sidebar.button("Estimar precio"):
#    st.write("¡Botón clickeado!")

# Añadir un menú desplegable a la barra lateral
#opcion_menu = sidebar.selectbox("Menú desplegable", ["Opción 1", "Opción 2", "Opción 3"])
#st.write(f"Opción seleccionada: {opcion_menu}")

# Añadir un cuadro de selección a la barra lateral
#opcion_cuadro = sidebar.selectbox("Cuadro de selección", ["Opción A", "Opción B", "Opción C"])
#st.write(f"Cuadro de selección: {opcion_cuadro}")

# Añadir una barra de deslizamiento para valores numéricos
#valor_slider = sidebar.slider("Slider", min_value=0, max_value=100, value=50)
#st.write(f"Valor del slider: {valor_slider}")

# Añadir un campo de entrada de texto a la barra lateral
#texto_input = sidebar.text_input("Entrada de texto", "Escribe aquí...")
#st.write(f"Texto ingresado: {texto_input}")

#===============================================================
#                     SECCIÓN IMAGEN
#===============================================================
#st.title('MODERN REAL ESTATE')
image = Image.open('img_1584_x_396_px.jpeg')
nuevo_tamano_img=(960,300)
image_redimensionada=image.resize(nuevo_tamano_img)
st.image(image_redimensionada, caption='@Empresa Inmobiliaria')
st.header("Descubre tu hogar, construye tu futuro: nuestro compromiso es tu felicidad.")
#===============================================================
#                  SECCIÓN BARRA LATERAL
#===============================================================
sidebar = st.sidebar
# Añadir un título a la barra lateral
sidebar.title("CALCULA EL PRECIO DE TU CASA")
st.image('MarcaEMPRESA ML-JPG.jpg',width=100)
#sidebar.header("Modelo Machine Learnign")

# Añadir un botón a la barra lateral
#if sidebar.button("Estimar precio"):
#    st.write("¡Botón clickeado!")

# Añadir un menú desplegable a la barra lateral
estilo_vivienda= sidebar.selectbox("ESTILO DE LA VIVIENDA", ["Un piso", "Un piso y medio: 2do nivel terminado", "Un piso y medio: 2do nivel sin terminar","Dos pisos","Dos pisos y medio: 2do nivel terminado","Dos pisos y medio: 2do nivel sin terminar"])
#st.write(f"Estilo de vivienda: {estilo_vivienda}")

# Añadir un cuadro de selección a la barra lateral
distancia = sidebar.text_input(label='DISTANCIA AV.PRINCIPAL-DOMICILIO (m)',value=0)
#st.write(f"Área habitable: {area_habitable}")

# Añadir una barra de deslizamiento para valores numéricos
numero_habitaciones = sidebar.slider("NÚMERO DE HABITACIONES", min_value=2, max_value=14, value=2)
#st.write(f"Número de habitaciones: {numero_habitaciones}")

# Añadir un campo de entrada de texto a la barra lateral
superficie_terreno = sidebar.text_input("SUPERFICIE DEL TERRENO (m2)", value=0)
#st.write(f"Superficie del terreno: {superficie_terreno}")

superficie_garaje = sidebar.text_input("SUPERFICIE DEL GARAJE (m2)", value=0)
#st.write(f"Superficie del garaje: {superficie_garaje}")


#=========================================================================
#           RESTRICCIONES
#=========================================================================

if sidebar.button("Estimar precio"):
    
  try:
      lot_Fr=float(distancia)
      Tot_hab=int(numero_habitaciones)
      lot_area=float(superficie_terreno)
      g_area=float(superficie_garaje)
      if (lot_area<=0 or Tot_hab<=0 ):
         st.warning('CAMPOS: SUPERFICIE DEL TERRENO! "INGRESAR NÚMERO POSITIVO" ', icon="🚨")
         if (lot_Fr<0 or g_area<0):
             st.warning('CAMPOS: DISTANCIA AV.PRINCIPAL-DOMICILIO & SUPERFICIE DEL GARAJE! "INGRESAR NÚMEROS MAYORES O IGUALES A CERO"', icon="🚨")
      else:
              new_data={'LotFrontage':[lot_Fr] ,
                         'LotArea':[lot_area],
                         'TotRmsAbvGrd':[Tot_hab],
                         'GarageArea':[g_area],
                         }
              new_data=pd.DataFrame(new_data)

              prediction= loaded_model.predict(new_data)
              print('Ejecutando predicción...')
              st.success(f' PRECIO ESTIMADO:  {prediction}', icon="🏡")
              
  except ValueError: 
     st.error('Los campos solo permiten ingresar números', icon="🚨") 
     st.info(f' NOTA: Si desea que el domicilo se ubique en la avenida principal escribir en DISTANCIA AV.PRINCIPAL-DOMICILIO ( 0 )',icon="ℹ️") 
     st.info(f' NOTA: Si desea que el domicilio no incluya garaje escribir en SUPERFICIE DEL GARAJE ( 0 )',icon="ℹ️") 
else:
      st.info(f' EL PRECIO ESTIMADO SE MOSTRARÁ AQUÍ ',icon="👉")     









