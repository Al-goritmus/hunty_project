# hunty_project
hunty_project


### 1)preprocess_sources_defintive.py

Este código es una solución completa para la tarea de preprocesamiento y modelado de recomendación de trabajos. Comienza con la importación de las bibliotecas necesarias y la carga de los datos de dos hojas de Excel que contienen información sobre las vacantes y los usuarios.

Para cada conjunto de datos, se realiza una breve exploración para comprender la estructura de los datos. A continuación, el código preprocesa y tokeniza los datos textuales de interés, como el nombre del rol deseado por el usuario y el nombre de la vacante. Para ello, se utilizan técnicas como la lematización y la eliminación de caracteres no alfanuméricos.

Para modelar las similitudes entre los roles deseados por los usuarios y los nombres de las vacantes, se utiliza un enfoque de factorización de la matriz TF-IDF, seguido de una medida de similitud del coseno. Esto permite encontrar las mejores correspondencias entre los roles deseados de los usuarios y las vacantes disponibles.

Además, el código implementa la sumarización del texto utilizando BERT para generar descripciones más breves de las vacantes. Esto se utiliza para proporcionar una visión más concisa de la vacante recomendada.

Por último, el código reformatea las recomendaciones generadas en un formato de prompt-completitud y las guarda en un archivo jsonl, que es una representación más eficiente para la siguiente fase de modelado con GPT. En el proceso, se asegura de eliminar los diálogos con alta similitud para evitar redundancias.

2)main3.py
Este codigo implementa un modelo simple para recomendaciones únicas. Los resultados para la función de costo y el accuracy están consignados en results.csv

### 3)top_5_recomendaciones.xlsx.  Este es un producto específico luego de realizar match entre las dos fuentes (vacantes y usuarios) relaciona datos relevantes de usuarios con datos relevantes de vacantes, es el producto principal de la ejecución del código consignado en preprocess_sources_definitive.

4)top_5_recomender.py
Este código implementa un modelo más complejo con una relación de aproximadamente 1:4 a 1:5 entre rol deseado y vacante para recomendar más de una vacante, en algunos casos 4 o 5 vacantes por usuario. Los resultados para la función de costo y el accuracy están consignados en results_2nd_model.csv
