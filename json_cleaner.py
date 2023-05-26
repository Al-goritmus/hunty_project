import json

# cargar data del jsonl
with open('primer_entrenamiento_reducido2.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]


def clean_data(example):
    if example['completion'][0]['summary_description'] == "":
        del example['completion'][0]['summary_description']
    return example

# limpieza de datos
cleaned_data = [clean_data(example) for example in data]

# almacenamiento de data modificada en jsonl
with open('cleaned_data.jsonl', 'w') as file:
    for example in cleaned_data:
        file.write(json.dumps(example) + '\n')
