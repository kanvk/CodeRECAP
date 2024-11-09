import torch
import faiss
import numpy as np
from uuid import uuid4
from transformers import AutoModel, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def vectorize_function(function_dict, tokenizer, model):
    # Extract the code from the function info
    code = function_dict["code"]  # Access the 'code' field

    # Tokenize the code
    inputs = tokenizer(
        code, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)  # Get the model outputs

    # Compute the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

    # Prepare the vectorized information as a dictionary
    vectorized_info = {
        "name": function_dict["name"],
        "path": function_dict["file_path"],  # Optional path if included
        "vector": embeddings,
    }

    return vectorized_info


def get_function_matches(embedding_query, vectorized_list, k):
    """
    Search for the top K most similar functions given a query embedding, using vectorized function info.
    """
    # Extract the code vectors and create a list of function names and paths
    code_vectors = np.vstack([func_info["vector"] for func_info in vectorized_list])

    dimension = code_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(code_vectors)

    # Search for the top K most similar vectors
    distances, indices = index.search(embedding_query.reshape(1, -1), k)

    # Retrieve top K results with function names, paths, and similarity scores
    results = [
        (
            vectorized_list[idx]["name"],
            vectorized_list[idx]["path"],
            1 / (1 + distances[0][i]),
        )
        for i, idx in enumerate(indices[0])
    ]

    return results


def unixcoder_embeddings(texts):
    # Convert a list of texts to embeddings using UniXcoder
    model_name = "microsoft/unixcoder-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Tokenize the list of texts
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
    )
    # Pass the inputs through the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # Use the last hidden state and average over tokens to get a single vector per text
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # Return as a numpy array
    return embeddings.numpy()


class UniXcoderEmbeddings(HuggingFaceEmbeddings):
    # Wrapping the unixcoder_embeddings() in the HuggingFaceEmbeddings class for use with LangChain
    def embed_text(self, text: str):
        return unixcoder_embeddings([text])[0]

    def embed_texts(self, texts: list):
        return unixcoder_embeddings(texts)

    def embed_documents(self, texts: list):
        # This method processes a list of documents (texts)
        return self.embed_texts(texts)

    def embed_query(self, text: str):
        # This method processes a single query
        return self.embed_text(text)


def create_documents_from_code_infos(code_infos):
    data = [
        (info["code_snippet"], {k: v for k, v in info.items() if k != "code_snippet"})
        for info in code_infos
    ]
    docs = [Document(page_content=doc[0], metadata=doc[1]) for doc in data]
    return docs


def save_to_faiss_vector_store(index_name, documents, embedding_model):
    # Create faiss index
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("print('hello world')")))
    # Create vector store with the index
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    uuids = [str(uuid4()) for _ in range(len(documents))]
    # add documents
    vector_store.add_documents(documents=documents, ids=uuids)
    # save
    vector_store.save_local(index_name)
    return vector_store


def load_faiss_vector_store(index_name, embedding_model):
    vector_store = FAISS.load_local(
        index_name, embedding_model, allow_dangerous_deserialization=True
    )
    return vector_store


def search_vector_store(vector_store, query, k):
    results = vector_store.similarity_search(query, k=k)
    return results
