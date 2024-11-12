import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

llama_model_name = "Meta-Llama-3.1-8B-Instruct"


def get_azure_chat_client():
    endpoint = "https://models.inference.ai.azure.com"
    token = os.getenv("AZURE_INFERENCE_CREDENTIAL")
    if not token:
        raise ValueError(
            "Please set the environment variable 'AZURE_INFERENCE_CREDENTIAL'."
        )
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )
    return client


def get_azure_llm_response(azure_client, model_name, user_prompt):
    try:
        response = azure_client.complete(
            messages=[
                SystemMessage(
                    content="You are a helpful assistant who is great at analyzing code repositories. You must answer all user queries by analyzing the factual information and code shared in the query."
                ),
                UserMessage(content=user_prompt),
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            model=model_name,
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            return "No response choices returned."
    except Exception as e:
        return f"Error occurred: {e}"
