# # import os
# # from azure.ai.inference import ChatCompletionsClient
# # from azure.ai.inference.models import SystemMessage, UserMessage
# # from azure.core.credentials import AzureKeyCredential

# # endpoint = "https://models.inference.ai.azure.com"
# # model_name = "Meta-Llama-3-8B-Instruct"
# # token = os.environ["GITHUB_TOKEN"]

# # client = ChatCompletionsClient(
# #     endpoint=endpoint,
# #     credential=AzureKeyCredential(token),
# # )

# # response = client.complete(
# #     messages=[
# #         SystemMessage(content="You are a helpful assistant."),
# #         UserMessage(content="What is the capital of France?"),
# #     ],
# #     temperature=1.0,
# #     top_p=1.0,
# #     max_tokens=1000,
# #     model=model_name
# # )

# # print(response.choices[0].message.content)

# # ------------------------------------
# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT License.
# # ------------------------------------
# """
# DESCRIPTION:
#     This sample demonstrates how to get text embeddings for a list of sentences
#     using a synchronous client. Here we use the service default of returning
#     embeddings as a list of floating point values.

#     This sample assumes the AI model is hosted on a Serverless API or
#     Managed Compute endpoint. For GitHub Models or Azure OpenAI endpoints,
#     the client constructor needs to be modified. See package documentation:
#     https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-inference/README.md#key-concepts

# USAGE:
#     python sample_embeddings.py

#     Set these two environment variables before running the sample:
#     1) AZURE_AI_EMBEDDINGS_ENDPOINT - Your endpoint URL, in the form 
#         https://<your-deployment-name>.<your-azure-region>.models.ai.azure.com
#         where `your-deployment-name` is your unique AI Model deployment name, and
#         `your-azure-region` is the Azure region where your model is deployed.
#     2) AZURE_AI_EMBEDDINGS_KEY - Your model key (a 32-character string). Keep it secret.
# """


# def sample_embeddings():
#     import os

#     try:
#         endpoint = os.environ["AZURE_AI_EMBEDDINGS_ENDPOINT"]
#         key = os.environ["AZURE_AI_EMBEDDINGS_KEY"]
#     except KeyError:
#         print("Missing environment variable 'AZURE_AI_EMBEDDINGS_ENDPOINT' or 'AZURE_AI_EMBEDDINGS_KEY'")
#         print("Set them before running this sample.")
#         exit()

#     # [START embeddings]
#     from azure.ai.inference import EmbeddingsClient
#     from azure.core.credentials import AzureKeyCredential

#     client = EmbeddingsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

#     response = client.embed(input=["first phrase", "second phrase", "third phrase"])

#     for item in response.data:
#         length = len(item.embedding)
#         print(
#             f"data[{item.index}]: length={length}, [{item.embedding[0]}, {item.embedding[1]}, "
#             f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
#         )
#     # [END embeddings]


# if __name__ == "__main__":
#     sample_embeddings()

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
        raise ValueError("Please set the environment variable 'AZURE_INFERENCE_CREDENTIAL'.")

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
            model=model_name
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {e}"

# GUI function
# def on_submit():
#     user_input = user_input_entry.get()
#     if user_input.strip() == "":
#         messagebox.showwarning("Input Error", "Please enter a question.")
#     else:
#         response_text = get_response(user_input)
#         output_text.delete(1.0, tk.END)
#         output_text.insert(tk.END, response_text)

# # Create the main window
# root = tk.Tk()
# root.title("Azure AI Assistant")
# root.geometry("600x400")

# # User input label and entry
# user_input_label = tk.Label(root, text="Enter your question:")
# user_input_label.pack(pady=10)

# user_input_entry = tk.Entry(root, width=80)
# user_input_entry.pack(pady=5)

# # Submit button
# submit_button = tk.Button(root, text="Get Response", command=on_submit)
# submit_button.pack(pady=10)

# # Output text area
# output_label = tk.Label(root, text="Assistant's Response:")
# output_label.pack(pady=10)

# output_text = tk.Text(root, wrap="word", height=10, width=70)
# output_text.pack(pady=5)

# # Start the Tkinter event loop
# root.mainloop()
