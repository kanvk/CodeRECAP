import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


# Function to set up LLaMA client
def setup_llama_client():
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "Meta-Llama-3-8B-Instruct"

    token = os.getenv("AZURE_INFERENCE_CREDENTIAL")

    if not token:
        raise ValueError(
            "Please set the environment variable 'AZURE_INFERENCE_CREDENTIAL'."
        )

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    return client, model_name


# Function to get a response from the LLaMA API
def get_response(client, model_name, user_input):
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=user_input),
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error occurred: {e}"
