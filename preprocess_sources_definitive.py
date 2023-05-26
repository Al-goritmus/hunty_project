# -*- coding: utf-8 -*-

"""
Este código tiene el fin de preprocesar el dataset de acuerdo con el entendimiento realizado en los EDAs a las fuentes y el objetivo de entrenamiento de un modelo de lenguaje natural
"""

##IMPORTAMOS LAS LIBRERÍAS PERTINENTES PARA EL ANÁLISIS Y LA GENERACIÓN DE UN ARCHIVO PARA ENTRENAR EL MODELO DE NLP.

#Librerías generales
import pandas as pd
import numpy as np
import matplotlib.pyplot     as plt
import matplotlib.patches    as mpatches
import seaborn               as sns
import sklearn.metrics       as Metrics
# %matplotlib inline

#Librerías de vectorización y similaridad para procesamiento de texto de manera vectorizada.
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


#delcarando part_dir y nombre de archivos
path_dir = '/home/men2a/hunty_project/sources/'
filename = 'vacantes.xlsx'
sheet = 'Sheet1'
filename2 = 'Data Set Users.xlsx'
sheet2 = 'Raw Data'

#data de vacantes
df_vacancies = pd.read_excel(f'{path_dir}{filename}', sheet_name=sheet, header=0)

#data de usuarios
df_users = pd.read_excel(f'{path_dir}{filename2}', sheet_name=sheet2, header=0)

#información general del dataset de vacantes
df_vacancies.info()

#información general del dataset de usuarios
df_users.info()

#contenido de la columna descripción
df_vacancies.description

#se verifica qué no hay coincidencia directa entre el nombre de la vacante y el rol qué desea ejecutar el usuario
df_vacancies.vacancy_name.isin(df_users.wish_role_name).sum()

#Tampoco hay coincidencias entre hardskills y nombre de vacantes
df_vacancies.vacancy_name.isin(df_users.hardskills).sum()


#€sta celda del código se usa para preprocesar y tokenizar el contenido de las columnas wish_role_name y vacancy_name

# se descarga el recurso "punkt"
nltk.download('punkt')

#esta función remueve puntuación => pasar a minúsculas => tokeniza palabras => identifica terminos vastago
def preprocess_text(text):
    if isinstance(text, str):
        text = text.replace('[^\w\s]', '') 
        text = text.lower()  
        tokens = word_tokenize(text)  
        stemmer = PorterStemmer() 
        stemmed_tokens = [stemmer.stem(token) for token in tokens] 
        preprocessed_text = ' '.join(stemmed_tokens)  
        return preprocessed_text
    else:
        return ""

# se aplica la función de preprocesamiento al texto de las columnas wish_role_name y vacancy_name
df_users['preprocessed_wish_role'] = df_users['wish_role_name'].apply(preprocess_text)
df_vacancies['preprocessed_vacancy_name'] = df_vacancies['vacancy_name'].apply(preprocess_text)

# se instancia un vectorizador utilizando TF-IDF
vectorizer = TfidfVectorizer()

# se hace fit_transform utilizando el vectorizador instanciado previamente sobre las columnas de texto preprocesado
X_users = vectorizer.fit_transform(df_users['preprocessed_wish_role'])
X_vacancies = vectorizer.transform(df_vacancies['preprocessed_vacancy_name'])


print(X_users.shape)
print(X_vacancies)


"""Prueba de sumarización """

import pandas as pd
from summa.summarizer import summarize

users_wish_role_column = 'preprocessed_wish_role'
vacancies_description_column = 'preprocessed_vacancy_name'


def text_summarization(text):
    summary_ratio = 0.8  # este valor establece un ratio entre el tamaño del texto original y la versión resumida
    summary = summarize(text, ratio=summary_ratio)
    return summary

# se aplica la sumarización a las columnas ['users_wish_role'] del dataframe df_users y ['preprocessed_vacancy_name']
df_users['summary_wish_role'] = df_users[users_wish_role_column].apply(text_summarization)
df_vacancies['summary_description'] = df_vacancies[vacancies_description_column].apply(text_summarization)

df_users['summary_wish_role'].value_counts()

df_vacancies['summary_description'].value_counts().head(50)

df_users[['wish_role_name','preprocessed_wish_role','summary_wish_role']].dropna().head(30)

"""La estrategia de summarize text no parece ser la mejor estrategia con la columna wish_role dada su corta longitud (len()) y la escasa cantidad de output valido incluso subiendo el ratio por encima de 0.5, se utilizará sólo para añadir una versión simplificada de la columna description al dataframe de vacantes, lo siguiente es hacer drop a la columna summary_wish_role """

df_users.drop('summary_wish_role',axis=1,inplace=True)

df_users.columns

df_vacancies.columns

"""producto de la implementación del fit_transform utilizando el vectorizador instanciado previamente sobre las columnas de texto preprocesado"""

X_users #matriz para comparación originada en wish_role

X_vacancies  #matriz originada en descripción

df_vacancies.to_excel('preprocessed_vacancies.xlsx')

df_users.to_excel('preprocessed_users.xlsx')

df_users.info()

"""Habiendo verificado las vectorización del texto para el trabajo deseado y el nombre de la vacante se puede proceder a implementar la similaridad coseno para comparar los vectores originados en las cadenas de texto, al calcular el 'ángulo' entre estos vectores se puede estimar desde un punto de vista de álgebra vectorial el grado de proximidad, similariad o parecido entre estas cadenas de texto, se sintetiza todo el proceso en un sólo código."""

# se aplica preprocesamiento a las columnas originales
df_users['preprocessed_wish_role'] = df_users['wish_role_name'].apply(preprocess_text)
df_vacancies['preprocessed_vacancy_name'] = df_vacancies['vacancy_name'].apply(preprocess_text)

# se instancia el vetorizador.
vectorizer = TfidfVectorizer()

# se implementa el vectorizador sobre las columnas de texto preprocesado.
X_users = vectorizer.fit_transform(df_users['preprocessed_wish_role'])
X_vacancies = vectorizer.transform(df_vacancies['preprocessed_vacancy_name'])


similarity_matrix = cosine_similarity(X_users, X_vacancies) # se calcula la similaridad coseno entre estos vectores. 

top_n_indices = np.apply_along_axis(lambda x: np.argpartition(x, -5)[-5:], 1, similarity_matrix) # se obtiene un top 5 de vacantes similares para cada usuario. 

recommendations_df = pd.DataFrame() #se crea un dataframe vacío dónde vamos a sintetizar la data correspondiente al entrenamiento del modelo.

lista_recomendaciones = []

# se itera sobre cada usuario y sus top de vacancia relativos originados en la matriz de similaridad usando np.argpartition con el fin de no aumentar la complejidad temporal del algoritmo.  
for user_index, vacancy_indices in enumerate(top_n_indices):
    user_id = df_users.loc[user_index, 'id_user']
    user_role = df_users.loc[user_index, 'wish_role_name']

    for vacancy_index in vacancy_indices:
        vacancy_name = df_vacancies.loc[vacancy_index, 'vacancy_name']
        vacancy_description = df_vacancies.loc[vacancy_index, 'description']
        vacancy_summary = df_vacancies.loc[vacancy_index, 'summary_description']
        vacancy_work_modality = df_vacancies.loc[vacancy_index, 'work_modality']
        
        # Add a dictionary with relevant vacancy data to the list
        lista_recomendaciones.append({
            'id_user': user_id,
            'wish_role_name': user_role,
            'vacancy_name': vacancy_name,
            'description': vacancy_description,
            'summary_description': vacancy_summary,
            'work_modality': vacancy_work_modality
        })

        #se constriiye el dataframe haciendo un append con las variables previamente declaradas con información relevante de las  vacantes 
        #recommendations_df = recommendations_df.append({
        #    'id_user': user_id,
        #    'wish_role_name': user_role,
        #    'vacancy_name': vacancy_name,
        #    'description': vacancy_description,
        #    'summary_description': vacancy_summary,
        #    'work_modality': vacancy_work_modality
        #}, ignore_index=True)
        
recommendations_df = pd.DataFrame(lista_recomendaciones)

# se imprime el top 5 de recomendaciones para cada usuario ahora condensado en un sólo dataframe
print(recommendations_df)

recommendations_df.info() #primera visión del dataset de entrenamiento

recommendations_df.head(50) #primeras 50 filas del dataset. Se evidencia la necesidad de manejar los valores nulos en la columna wish_role_name, qué no superan el 20% de los valores

recommendations_df.dropna(subset='wish_role_name') #dado qué esta columna funcionará cómo el prompting para el modelo e igual en el entrenamiento, se elimina las filas en las cual este valor es nulo
#Pandas improme una coipia del dataframe cuando no usamos df = df... o inplace=True.

recommendations_df.dropna(subset=['wish_role_name'],inplace=True)

print('recomendations_df',recommendations_df.head(10))

recommendations_df.to_excel('dataset_entrenamiento1.xlsx')

"""#Por ultimo, sabemos qué los modelos GPT cómo el qué vamos a implementar en este proyecto son modelos qué reciben de manera preferencial versiones tokenizadas de texto en un formato  prompt + completion y para este fin el formato json es el ideal, se procede a correr código en python con el fin de generar un json formateado con el contenido """

import pandas as pd
import json

# leer el dataset de recomendaciones en caso de utilizar este cómo fuente.
# df = pd.read_excel('data_entrenamiento1.xlsx')  # en caso de utilizar un dataset qué fue previamente almacenado en un exce, es decir si recommendations_df es un excel.

# Create a list of dictionaries with the desired format
data = []
for _, row in recommendations_df.iterrows():
    # considerando el uso de 'wish_role_name' cómo prompt
    prompt = row['wish_role_name']
    
    # este ciclo for concatena las variables contenidas en las columnas ['vacancy_name', 'description', and 'summary_description'] as completion
    # se añaden tokens especiales para indicar al modelo el inicio y final de cada campo 
    completion = "<vacancy_name_start> " + row['vacancy_name'] + " <vacancy_name_end> " + \
                 "<description_start> " + row['description'] + " <description_end> " + \
                 "<summary_description_start> " + row['summary_description'] + " <summary_description_end>"
    data.append({"prompt": prompt, "completion": completion})

# El ciclo for anterior genera una lista de diccionarios ahora los incluimos en un JSON con encoding = 'Unicode'
with open('primer_entrenamiento.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

