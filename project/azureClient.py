import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


load_dotenv()


AZURE_INFERENCE_CREDENTIAL = os.getenv("AZURE_INFERENCE_CREDENTIAL")
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o"


def get_azure_chat_client():
    client = ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(AZURE_INFERENCE_CREDENTIAL),
    )
    return client


def get_azure_llm_response(azure_client, model_name, user_prompt):
    try:
        response = azure_client.complete(
            messages=[
                SystemMessage(
                    content="You are an expert code assistant for analyzing, summarizing, and locating elements within code repositories. Follow the user's instructions exactly, using only the provided information to deliver precise, concise answers. Avoid creating new information or making assumptions beyond the context given."
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
