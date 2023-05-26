import openai
import json

openai.api_key="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"

# Function to store user information and conversation history in a dictionary
def store_user_data(user_id, conversation, user_response=None):
    if user_response is not None:
        conversation += "\n" + user_response
    user_data = {
        user_id: {
            "user_id": user_id,
            "conversation": conversation,
        }
    }
    return user_data

# Function to generate response using conversation history
def generate_response(username, user_data, top_p=0.1, n=1, frequency_penalty=0.1):
    conversation_history = user_data[username]["conversation"]
    prompt = conversation_history + "\n\n###\n\n"
    response = openai.Completion.create(
        model='curie:ft-personal-2023-05-26-14-25-26', 
        max_tokens=350,
        prompt=prompt,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty,
        stop='###'
    )

    response_text = response.choices[0].text.strip()
    return response_text

def main():
    user_data = {}
    user_id = input("Ingrese su identificación de usuario: ")
    while True:
        prompt = input("Ingrese el nombre de su rol deseado: ")
        user_data = store_user_data(user_id, user_id, prompt)
        response = generate_response(user_id, user_data)
        print(response)
        user_data = store_user_data(user_id, prompt)

if __name__ == "__main__":
    main()
