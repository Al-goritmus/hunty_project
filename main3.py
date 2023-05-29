import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt, top_p=0.1, temperature=0.6, n=1, frequency_penalty=0.1):
    # Generar la respuesta usando la API 
    response = openai.Completion.create(
        model='curie:ft-personal-2023-05-26-21-36-50',
        max_tokens=150,
        prompt=prompt,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty,
        stop=[".0"] 
    )

    response_text = response.choices[0].text.strip()
    return response_text

def format_response(response_text):
    # Dividir la respuesta en sus componentes
    parts = response_text.split(',')

    # Verificar que cada trabajo tenga 4 partes
    if len(parts) % 4 != 0:
        return "The model's response didn't fit the expected format. Please try again."

    # formatear respuesta
    formatted_response = "Based on your input, here are some job suggestions for you:\n"
    for i in range(0, len(parts), 4):
        formatted_response += f"{(i//4) + 1}. Job Title: {parts[i]}\n"
        formatted_response += f"   Description: {parts[i+1]}\n"
        formatted_response += f"   Work Modality: {parts[i+2]}\n"
        formatted_response += f"   Job ID: {parts[i+3]}\n\n"
    return formatted_response


def main():
    while True:
        prompt = input("Ingrese el nombre de su rol deseado: ")
        response = generate_response(prompt)
        formatted_response = format_response(response)
        print(response)

if __name__ == "__main__":
    main()

