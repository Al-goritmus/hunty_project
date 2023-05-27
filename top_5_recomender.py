import openai

openai.api_key = "API_KEY"

def generate_response(prompt, top_p=0.1, temperature=0.8, n=1, frequency_penalty=0.1):
    # Generate the response
    response = openai.Completion.create(
        model='curie:ft-personal-2023-05-26-21-57-59',
        max_tokens=150,
        prompt=prompt,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty,
        stop=[".0}]"] #`stop=[".0"]` 
    )

    response_text = response.choices[0].text.strip()
    return response_text

def format_response(response_text):
    # Split the response into its parts
    parts = response_text.split(',')

    # Construct formatted response
    formatted_response = "Based on your input, here are some job suggestions for you:\n"
    formatted_response += f"1. Job Title: {parts[0]}\n"
    if len(parts) > 1:
        formatted_response += f"2. Job Title: {parts[1]}\n"
        formatted_response += f"   Job ID: {parts[-1]}\n"
    return formatted_response

def main():
    while True:
        prompt = input("Ingrese el nombre de su rol deseado: ")
        response = generate_response(prompt)
        formatted_response = format_response(response)
        print(formatted_response)

if __name__ == "__main__":
    main()

