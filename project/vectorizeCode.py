from transformers import RobertaTokenizer, RobertaModel
import torch
import faiss
import numpy as np

def vectorize_function(function_dict, tokenizer, model):
    # Extract the code from the function info
    code = function_dict['code']  # Access the 'code' field
    
    # Tokenize the code
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)  # Get the model outputs
    
    # Compute the embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    
    # Prepare the vectorized information as a dictionary
    vectorized_info = {
        'name': function_dict['name'],
        'path': function_dict['file_path'],  # Optional path if included
        'vector': embeddings
    }
    
    return vectorized_info

def get_function_matches(embedding_query, vectorized_list, k):
    """
    Search for the top K most similar functions given a query embedding, using vectorized function info.
    """
    # Extract the code vectors and create a list of function names and paths
    code_vectors = np.vstack([func_info['vector'] for func_info in vectorized_list])
    
    dimension = code_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(code_vectors)
    
    # Search for the top K most similar vectors
    distances, indices = index.search(embedding_query.reshape(1, -1), k)
    
    # Retrieve top K results with function names, paths, and similarity scores
    results = [
        (vectorized_list[idx]['name'], vectorized_list[idx]['path'], 1 / (1 + distances[0][i]))
        for i, idx in enumerate(indices[0])
    ]
    
    return results

def vectorize_code(code, tokenizer, model):
    # Tokenize the code using UniXcoder's tokenizer
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get the code embedding using the model
    with torch.no_grad():
        outputs = model(**inputs)

    # The embeddings are in the 'last_hidden_state'
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pool to get a single vector

    return embeddings.numpy()  # Return as NumPy array


def get_matches(embeddings_query, code_vectors  , k):
    code_vectors = np.vstack(code_vectors)
    
    dimension = code_vectors.shape[1]
    
    index = faiss.IndexFlatL2(dimension)

    # Add the code vectors to the FAISS index
    index.add(code_vectors)
    
    # Search for the top K most similar vectors
    distances, indices = index.search(embeddings_query.reshape(1, -1), k)  # Reshape for a single query

    file_names = [i for i in range(1, len(code_vectors) + 1)]

    results = [(file_names[idx], 1 / (1 + distances[0][i])) for i, idx in enumerate(indices[0])]  # Use inverse distance for similarity

    return results
