import os
from zipfile import ZipFile
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
import joblib

#Descargamos la carpeta en formato zip y exploramos los archivos que guarda
dataset_url = 'https://github.com/JovianML/opendatasets/raw/master/data/house-prices-advanced-regression-techniques.zip'
urlretrieve(dataset_url, 'house-prices.zip')
with ZipFile('house-prices.zip') as f:
    f.extractall(path='house-prices')
os.listdir('house-prices')
print('Los archivos encontrados en la carpeta zip "house-prices" son:', os.listdir('house-prices'))

# número máximo de filas que se mostrarán al imprimir un DataFrame
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20

#el DataFrame prices_df contendrá los datos del archivo .CSV
prices_df = pd.read_csv('house-prices/train.csv')

# Identify input and target columns
input_cols, target_col =['LotFrontage','LotArea','TotRmsAbvGrd','GarageArea'], prices_df.columns[-1] #Se crean listas solo con los nombres de las columnas de dataFramee
#inputs_df, targets = prices_df[input_cols].copy(), prices_df[target_col].copy() #se crea dataframes solo con los nombres de las columnas
inputs_df, targets = prices_df[['LotFrontage','LotArea','TotRmsAbvGrd','GarageArea']], prices_df[target_col].copy()
print(inputs_df)
print(targets)

#identificamos columnas númericas y columnas categoricas 
numeric_cols = prices_df[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = prices_df[input_cols].select_dtypes(include='object').columns.tolist()

#Hacemos un cambio de unidades de las variables área, de ft2 a m2 
ft2=0.092903 #Comstante que nos permite convertor de ft2 a m2
valores1=inputs_df['LotArea']
inputs_df['LotArea']=[round(x*ft2) for x in valores1]
valores2=inputs_df['GarageArea']
inputs_df['GarageArea']=[round(x*ft2) for x in valores2]
#hacemos un cambio de unidades de ft a m
ft=0.304804
valores3=inputs_df['LotFrontage'].fillna(0)
inputs_df['LotFrontage']=[round(x*ft) for x in valores3]
print(inputs_df)

#crear conjuntos de Entrenamiento y validación
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs_df[numeric_cols], targets, test_size=0.2, random_state=42)
print(train_inputs)
print('¡Esta hecho!')

#Implementamos el modleo de regresión de árbol de decisión decisionTreeRegressor

# Create the model
model = DecisionTreeRegressor()
print('¡Esta creado el modelo!')

#Ajustamos el modelo a los datos de entrenamiento 
model.fit(train_inputs,train_targets)
print('¡Modelo Entrenado!')

#Checamos la importancia de las variables 
# Check feature importance
tree_importances = model.feature_importances_
tree_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': tree_importances
}).sort_values('importance', ascending=False)
print('Importancia de las variables del modelo')
print(tree_importance_df)

#Evaluamos el rendimiento del modelo
# Hacer predicciones sobre el conjunto de prueba
predictions = model.predict(train_inputs)
# Evaluar el rendimiento del modelo
mse = mean_squared_error( predictions,train_targets)
r2 = r2_score(train_targets, predictions)
print("Error cuadrático medio MSE (Test):", mse)
print("Valor R² (Test):", r2)

#Guardar y Cargar el modelo
#pickle
model_filename = './house_prices.pkl'
joblib.dump(model, model_filename)
print("Modelo grabado!")
