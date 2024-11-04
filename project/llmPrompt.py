import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


# Function to set up LLaMA client
def setup_llama_client():
    """
    Sets up the LLaMA client using Azure's ChatCompletionsClient.

    Returns:
        tuple: A tuple containing the client and the model name.
    """
    endpoint = "https://models.inference.ai.azure.com"
    model_name = "Meta-Llama-3-8B-Instruct"
    
    token = os.getenv("AZURE_INFERENCE_CREDENTIAL")
    
    if not token:
        raise ValueError("Please set the environment variable 'AZURE_INFERENCE_CREDENTIAL'.")

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    return client, model_name

# Function to get a response from the LLaMA API
def get_response(client, model_name, user_input):
    """
    Sends a user input message to the LLaMA model and retrieves the response.

    Args:
        client: The ChatCompletionsClient instance.
        model_name: The name of the model to use.
        user_input: The input message from the user.

    Returns:
        str: The assistant's response or an error message.
    """
    try:       
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=user_input),
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            return "No response choices returned."
        
    except Exception as e:
        return f"Error occurred: {e}"

def summarize_readme(clone_dir, client, model_name):
    """
    Summarizes the README file in the specified directory.

    Args:
        clone_dir: The path to the cloned repository.
        client: The ChatCompletionsClient instance.
        model_name: The name of the model to use.

    Returns:
        str: The summary of the README file or a message if not found.
    """
    readme_path = os.path.join(clone_dir, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        prompt = f"Summarize the following README file:\n\n{readme_content}"
        summary = get_response(client, model_name, prompt)
        return summary
    else:
        return "No README.md file found in the repository."
