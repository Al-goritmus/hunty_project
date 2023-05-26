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


#€sta parte del código se usa para preprocesar y tokenizar el contenido de las columnas wish_role_name y vacancy_name

# se descarga el recurso "punkt"
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):
        text = text.replace('[^\w\s]', '') 
        text = text.lower()  
        tokens = word_tokenize(text) 
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens] 
        preprocessed_text = ' '.join(lemmatized_tokens)  
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


#def text_summarization(text):
#    summary_ratio = 0.8  # este valor establece un ratio entre el tamaño del texto original y la versión resumida
#    summary = summarize(text, ratio=summary_ratio)
#    return summary

# se aplica la sumarización a las columnas ['users_wish_role'] del dataframe df_users y ['preprocessed_vacancy_name']
#df_users['summary_wish_role'] = df_users[users_wish_role_column].apply(text_summarization)
#df_vacancies['summary_description'] = df_vacancies[vacancies_description_column].apply(text_summarization)


#df_users[['wish_role_name','preprocessed_wish_role','summary_wish_role']].dropna().head(30)

"""La estrategia de summarize text no parece ser la mejor estrategia con la columna wish_role dada su corta longitud (len()) y la escasa cantidad de output valido incluso subiendo el ratio por encima de 0.5, se utilizará sólo para añadir una versión simplificada de la columna description al dataframe de vacantes, lo siguiente es hacer drop a la columna summary_wish_role """


from summarizer import Summarizer

users_wish_role_column = 'preprocessed_wish_role'
vacancies_description_column = 'preprocessed_vacancy_name'


# se cargar el BERT pre entrenado
model = Summarizer()

def bert_summarization(text):
    # El ratio especifica la información a retener 
    ratio = 0.8
    summary = model(text, ratio=ratio)
    return summary

# Se aplica la sumarización a las columnas deseadas
df_users['summary_wish_role'] = df_users[users_wish_role_column].apply(bert_summarization)
df_vacancies['summary_description'] = df_vacancies[vacancies_description_column].apply(bert_summarization)

df_users[['wish_role_name','preprocessed_wish_role','summary_wish_role']].dropna().head(30)


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
        #vacancy_description = df_vacancies.loc[vacancy_index, 'description']
        vacancy_summary = df_vacancies.loc[vacancy_index, 'summary_description']
        vacancy_work_modality = df_vacancies.loc[vacancy_index, 'work_modality']
        vacancy_account_executive = df_vacancies.loc[vacancy_index, 'account executive']
        
        # Add a dictionary with relevant vacancy data to the list
        lista_recomendaciones.append({
            'id_user': user_id,
            'wish_role_name': user_role,
            'vacancy_name': vacancy_name,
            #'description': vacancy_description,
            'summary_description': vacancy_summary,
            'work_modality': vacancy_work_modality,
            'account executive': vacancy_account_executive
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

#recommendations_df.dropna(subset='wish_role_name') #dado qué esta columna funcionará cómo el prompting para el modelo e igual en el entrenamiento, se elimina las filas en las cual este valor es nulo
#Pandas improme una coipia del dataframe cuando no usamos df = df... o inplace=True.

recommendations_df.dropna(subset=['wish_role_name'],inplace=True)

print('recomendations_df',recommendations_df.head(10))

recommendations_df.to_excel('top_5_recomendaciones.xlsx') #aquí se genera luego de procesada la data y hacer un match apropiado, un excel con el top de recomendaciones por usuario

import json

# Define a function to convert each row to a prompt-completion format
def convert_to_prompt_completion(record):
    # Extract the values as strings
    wish_role_name = str(record['wish_role_name'])
    vacancy_name = str(record['vacancy_name'])
    summary_description = str(record['summary_description'])
    work_modality = str(record['work_modality'])
    account_executive = str(record['account executive'])
    
    # Create the prompt and completion
    prompt = wish_role_name
    completion = vacancy_name + ', ' + summary_description + ', ' + work_modality + ', ' + f'Job ID: {account_executive}'

    # Return as a dict
    return {'prompt': prompt, 'completion': completion}

# Convert each row to the prompt-completion format
converted_data = [convert_to_prompt_completion(row) for _, row in recommendations_df.iterrows()]

# Save converted data to a jsonl file
with open('converted_data.jsonl', 'w') as file:
    for record in converted_data:
        file.write(json.dumps(record) + '\n')



"""#Por ultimo, sabemos qué los modelos GPT cómo el qué vamos a implementar en este proyecto son modelos qué reciben de manera preferencial versiones tokenizadas de texto en un formato  prompt + completion y para este fin el formato json es el ideal, se procede a correr código en python con el fin de generar un json formateado con el contenido """

import pandas as pd
import json

# leer el dataset de recomendaciones en caso de utilizar este cómo fuente.
# df = pd.read_excel('data_entrenamiento1.xlsx')  # en caso de utilizar un dataset qué fue previamente almacenado en un exce, es decir si recommendations_df es un excel.

# se añade un string vacío de base
recommendations_df['summary_description'] = recommendations_df['summary_description'].fillna('No summary description provided.')

data = []
for _, row in recommendations_df.iterrows():  
    prompt = row['wish_role_name']
    
    completion = [{"vacancy_name": row['vacancy_name'], 
                   "summary_description": row['summary_description'], 
                   "account executive": row['account executive']}]
    data.append({"prompt": prompt, "completion": completion})

# Convert list of dictionaries into a DataFrame
recommendations_df = pd.DataFrame(data)

# create a new 'conversation' column
recommendations_df['conversation'] = recommendations_df['prompt'] + ' ' + recommendations_df['completion'].apply(json.dumps)




# se vectoriza la columna de conversación
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(recommendations_df['conversation'])

# se aplica la similaridad coseno usando pairwise de scikit-learn
similarity_matrix = cosine_similarity(X)

# Se fija un humbral de similaridad
similarity_threshold = 0.8

# se remueven aquellos dialogos qué presenten una elevada similaridad coseno
unique_conversation_indices = []
for i in range(similarity_matrix.shape[0]):
    if not any(similarity_matrix[i, :i] > similarity_threshold):
        unique_conversation_indices.append(i)

unique_recommendations_df = recommendations_df.iloc[unique_conversation_indices]


# se genera el archivo JSON
with open('primer_entrenamiento_reducido2.jsonl', 'w', encoding='utf-8') as f:
    for _, row in unique_recommendations_df.iterrows():
        item = {"prompt": row['prompt'], "completion": row['completion']}
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

