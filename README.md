##README

Instalación de librerías: pip install -r requirements.txt del archivo 'requirements.txt'

Este código es una solución completa para la tarea de preprocesamiento y modelado de recomendación de trabajos. Comienza con la importación de las bibliotecas necesarias y la carga de los datos de dos hojas de Excel que contienen información sobre las vacantes y los usuarios.

Para cada conjunto de datos, se realiza una breve exploración para comprender la estructura de los datos. A continuación, el código preprocesa y tokeniza los datos textuales de interés, como el nombre del rol deseado por el usuario y el nombre de la vacante. Para ello, se utilizan técnicas como la lematización y la eliminación de caracteres no alfanuméricos.

Para modelar las similitudes entre los roles deseados por los usuarios y los nombres de las vacantes, se utiliza un enfoque de factorización de la matriz TF-IDF, seguido de una medida de similitud del coseno. Esto permite encontrar las mejores correspondencias entre los roles deseados de los usuarios y las vacantes disponibles.

Además, el código implementa la sumarización del texto utilizando BERT para generar descripciones más breves de las vacantes. Esto se utiliza para proporcionar una visión más concisa de la vacante recomendada.

Por último, el código reformatea las recomendaciones generadas en un formato de prompt-completitud y las guarda en un archivo jsonl, que es una representación más eficiente para la siguiente fase de modelado con GPT. En el proceso, se asegura de eliminar los diálogos con alta similitud para evitar redundancias.

Archivos principales: 

1)preprocess_sources_definitive.py   Código principal del repositorio, utiliza las librerías mencionadas para generar datasets preprocesados y el top_5_recomendaciones en excel

2)top_5_recomendaciones.xlsx    Esta fuente se procesa y se convierte en cleaned_data.jsonl.

3)cleaned_data_prepared.jsonl   

4)results.csv (primer modelo) u results_2nd_model.csv (accuracy y función de costo del segundo modelo)

5)top_5_recomender.py implementación del modelo de multiples recomendaciones con formato en python

Entre otros archivos hay json de entrenamiento para modelos, notebooks de análisis de datos y el notebook preprocess_sources_definitive.ipynb qué permitió la prueba de todas las librerías implementadas en el proyecto.
