import os
import ast
import shutil
import hashlib
import requests
import numpy as np
import pandas as pd
from git import Repo
from pathlib import Path
from collections import Counter
import time
from datasets import load_dataset
import openai
from openai import OpenAI
import tiktoken
from langchain.prompts import PromptTemplate
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# The code in this file is used to get the derived dataset from SWE bench and perform LLM Prompting and Generate embeddings for each file

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the UniXcoder model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base").to(device)  # Move the model to GPU if available

tokenizer = tiktoken.get_encoding("cl100k_base")

# Set your OpenAI API key
openai.api_key = userdata.get('OPENAI_API_KEY')
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# Prompt for LLM Call
code_location_template = PromptTemplate(
    input_variables=[
        "repo_name",
        "files_list",
        "query",
    ],
    template="""
You are provided with details about the repository named **{repo_name}**.

**List of Files in the Repository:**
{files_list}

**User Query:**
{query}

**Your Task**:
- Identify the exact file (or files) names that needs to be modified to solve the problem described in the user query based on the files list provided.

**Response Format**:
- Return a json string with the below format
- If you are unable to determine the answer, respond with {{"answered": False}}
- If you are able to determine the answer, respond with {{"answered": True, "num_files": <integer>, "files": {{"rank_1": "file path", "rank_2": "file path", ...}}}} and rank the files in the order of most likely to be needed to be modified to solve the **user query**. You can choose upto 10 files. You can choose less than 10 files as well but need to select at least one file if you set "answered":True. The number of files you choose must be set in "num_files" in the response json and the selected files must be ranked and returned as "files": {{"rank_1": "file path", "rank_2": "file path", ...}} in the response json.

**CRITICAL INSTRUCTIONS**:
1. **DO NOT create or assume any file names**. Only return file names or files paths if it matches exactly with one from the above list of **files in the repository**.
2. **Respond ONLY with the json format described**. Avoid explanations, additional text, or clarifications.
3. **If the query cannot be answered with the provided information**, respond strictly with: {{"answered": False}}
"
**IMPORTANT**: Precision is essential; ensure that your answer is concise and follows the format exactly.
""",)


def get_modified_files(patch: str):
    """Gets the list of modified files from a patch.

    Args:
        patch (str): git diff patch

    Returns:
        list[str]: Modified files
    """
    modified_files = []
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) > 2:
                # The third part is the file path prefixed with 'a/'
                modified_file = parts[2].replace("a/", "", 1)
                modified_files.append(modified_file)
    return modified_files


def get_swebench_dataset() -> pd.DataFrame:
    """Loads the SWE-bench dataset."""
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")
    df = pd.DataFrame(ds["test"])
    df["modified_files"] = df["patch"].apply(get_modified_files)
    df = df[
        ["repo", "base_commit", "modified_files",
            "problem_statement", "hints_text"]
    ]
    return df

def generate_string_hash(string, hash_algorithm='sha256'):
    # Given a string (file contents) return a sha256 hash
    return hashlib.new(hash_algorithm, string.encode('utf-8')).hexdigest()

def clone_repository(repo_url, branch=None, commit_hash=None):
    # Clone the repo with url=repo_url into the cloned_repos directory
    # If repo already exists under cloned_repos, run git pull instead to update it
    repo_name = repo_url.split("/")[-1]
    repo_name = repo_name[:-4] if repo_name.endswith(".git") else repo_name
    clone_dir = f"./cloned_repos/{repo_name}"
    try:
        # If the directory exists, perform git pull, else clone it
        if os.path.exists(clone_dir):
            repo = Repo(clone_dir)
            try:
                repo.git.checkout("main") # if it was previously checked out to a random commit, switch to main and pull to update
            except:
                repo.git.checkout("master")
            repo.remotes.origin.pull()
        else:
            Repo.clone_from(repo_url, clone_dir)
        # if a branch is passed as arguments to the method, checkout to that branch
        if branch:
            repo = Repo(clone_dir)
            repo.git.checkout(branch)
        # if a commit_hash is passed as arguments to the method, checkout to that commit_hash
        if commit_hash:
            repo = Repo(clone_dir)
            repo.git.checkout(commit_hash)
        return repo_name, clone_dir
    except Exception as e:
        raise e

def remove_directory(cloned_dir):
    shutil.rmtree(cloned_dir)

def generate_chunk_embedding(code_chunk):
    # Generate embeddings for each chunk of code
    return openai.embeddings.create(
        model="text-embedding-3-small",
        input=code_chunk
    ).data[0].embedding

def aggregate_embeddings(embeddings):
    # Aggregate chunk embeddings
    return np.mean(embeddings, axis=0)

def num_tokens_from_string(string: str) -> int:
    # Return the number of tokens in a text string
    tokens = tokenizer.encode(string)
    return len(tokens)

def chunk_code_by_tokens(code_text, max_tokens=6000):
    # Tokenize the input code
    tokens = tokenizer.encode(code_text)
    # Split the tokens into chunks of max_tokens length
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        # Decode the chunk back into text/code
        chunk_text = tokenizer.decode(chunk)
        chunks.append(chunk_text)
    return chunks

def generate_file_embedding(file_string):
    # if less than 8000 tokens, get single embedding
    if num_tokens_from_string(file_string)<8000:
        return generate_chunk_embedding(file_string)
    # else chunk file, embed chunks, aggregate embeddings
    chunks = chunk_code_by_tokens(file_string)
    embeddings = [generate_chunk_embedding(chunk) for chunk in chunks]
    return aggregate_embeddings(embeddings)

def get_file_infos(clone_dir):
    """Analyze all Python files in the cloned repository, including all subdirectories."""
    file_infos = []
    for root, _, files in os.walk(clone_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding='utf-8') as f:
                        code = f.read().strip()
                        # Add all python files into a global list
                        if code:
                            file_infos.append(
                                {"file_path": Path(file_path).relative_to(clone_dir), "name": file,
                                    "code_snippet": code, "hash": generate_string_hash(code)}
                            )
                except Exception as e:
                    print(f" --- Error for file {file_path} {e}. Skipping file.")
    return file_infos

def format_repo_name(name):
    return "_".join(name.split("/"))

def make_request(prompt):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        temperature = 0.1,
        response_format = { "type": "json_object"},
        messages = [
            {
                "role": "system",
                "content": "You are an expert code assistant for analyzing, summarizing, and locating elements within code repositories. Follow the user's instructions exactly, using only the provided information to deliver precise, concise answers. Avoid creating new information or making assumptions beyond the context given."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content

def make_throttled_requests(df):
    # make a request every 20 seconds to keep withing rate limit
    for idx,row in df.iterrows():
        user_query = f"**Problem Description**:\n {row['problem_statement']}"
        if not pd.isna(row['hints_text']):
            user_query+=f"\n\n **Hint**: (You may use this hint if it is relevant or else ignore it) \n {row['hints_text']}\n"
        prompt = code_location_template.format(repo_name=row['repo'], files_list=row['files_in_repo'], query=user_query)
        res = make_request(prompt)
        with open("./data/results.jsonl", "a") as f:
            f.write(f"{res}\n")
        with open("./data/order.csv", "a") as f:
            f.write(f"{row['base_commit']},{idx}\n")
        print(f"- Processed Row {idx} - {row['base_commit']}")
        time.sleep(20)

def get_embeddings(text: str):
    # Tokenize the input text and convert it to input IDs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Move the tokenized inputs to GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Averaging the token embeddings
    return embeddings

def aggregate_embeddings(embeddings):
    # Aggregate embeddings from all chunks by averaging
    embeddings_tensor = torch.cat(embeddings, dim=0)
    aggregated_embedding = embeddings_tensor.mean(dim=0, keepdim=True)
    return aggregated_embedding

def get_document_embedding(document: str, chunk_size=512):
    # Split the document into smaller chunks if it exceeds the chunk size (token limit)
    chunks = [document[i:i + chunk_size] for i in range(0, len(document), chunk_size)]
    embeddings = []
    for chunk in chunks:
        embeddings.append(get_embeddings(chunk))
    final_embedding = aggregate_embeddings(embeddings)
    return final_embedding.squeeze().cpu().numpy().tolist()

def get_embeddings_file_path(repo_name):
    repo = "_".join(repo_name.split("/"))
    embeddings_file_path = f'./data/embeddings/{repo}_files_infos.csv'
    return embeddings_file_path

def search_top_k_docs(file_embeddings, query_embedding, k=15):
    index_cpu = faiss.IndexFlatIP(768)
    faiss.normalize_L2(file_embeddings)
    faiss.normalize_L2(query_embedding)
    index_cpu.add(file_embeddings)
    if device=='cuda':
        gpu_res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
        distances, indices = index_gpu.search(query_embedding, k)
    else:
        distances, indices = index_cpu.search(query_embedding, k)
    return distances[0], indices[0]

# Document vectorization
def vectorize_documents_tfidf(file_contents):
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(file_contents)
    return tfidf_matrix, vectorizer

# Query comparison
def search_tfidf(query, vectorizer, tfidf_matrix, file_names, k):
    query_tfidf = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    # Get file names in descending order of similarity without returning similarity scores
    results = sorted(zip(file_names, cosine_similarities), key=lambda x: x[1], reverse=True)
    ranked_file_names = [file_name for file_name, _ in results[:k]]
    return ranked_file_names

if __name__=="__main__":
    # Load SWE Bench Dataset
    swebench_df = get_swebench_dataset()
    # Group by repo name and get a list of commits for each repo
    repo_commits = swebench_df.groupby('repo')['base_commit'].apply(list).to_dict()
    # Clone at each base_commit and identify all the files at that commit and get the hashes for all files at that commit
    commit_to_files_map = {}
    commit_to_hashes_map = {}
    for repo in repo_commits:
        if repo.endswith("astropy") or repo.endswith("django") or repo.endswith("matplotlib"):
            print(f"Skipping {repo} as it has already been processed")
        print(f"Processing repo: {repo}")
        infos = []
        hashes = []
        for commit in repo_commits[repo]:
            print(f" - Processing commit {commit} in repo {repo}")
            repo_url = f"https://github.com/{repo}"
            repo_name, cloned_dir = clone_repository(repo_url=repo_url, commit_hash=commit)
            file_infos = get_file_infos(cloned_dir)
            for info in file_infos:
                if info['hash'] not in hashes:
                    hashes.append(info['hash'])
                    infos.append(info)
            commit_to_files_map[commit] = [info["file_path"] for info in file_infos]
            commit_to_hashes_map[commit] = [info["hash"] for info in file_infos]
            # remove_directory(cloned_dir)
        print(f" - Found {len(infos)} unique files in {repo} across {len(repo_commits[repo])} commits")
        df = pd.DataFrame(infos)
        info_csv_path = f'./data/{format_repo_name(repo)}_files_infos.csv'
        df.to_csv(info_csv_path, index=False)
        print(f"CSV file with all necessary file infos for repo {repo_name} has been saved to {info_csv_path}. Contains data on {len(infos)} files.")
    swebench_df['files_in_repo'] = swebench_df['base_commit'].apply(lambda x: [str(f) for f in commit_to_files_map[x]])
    swebench_df['hashes'] = swebench_df['base_commit'].apply(lambda x: commit_to_hashes_map[x])
    # write to file
    swebench_df.to_csv(f'./data/swebench_updated.csv', index=False)
    # For each row query the LLM to get LLM Prompting results
    make_throttled_requests(swebench_df)
    # For each file generate an embedding
    repo_csvs = [
        "astropy_astropy_files_infos.csv",          
        "django_django_files_infos.csv",            
        "matplotlib_matplotlib_files_infos.csv",    
        "mwaskom_seaborn_files_infos.csv",          
        "pallets_flask_files_infos.csv",            
        "psf_requests_files_infos.csv",             
        "pydata_xarray_files_infos.csv",            
        "pylint-dev_pylint_files_infos.csv",        
        "pytest-dev_pytest_files_infos.csv",        
        "scikit-learn_scikit-learn_files_infos.csv",
        "sphinx-doc_sphinx_files_infos.csv",        
        "sympy_sympy_files_infos.csv"               
    ]
    tqdm.pandas()
    for file_name in repo_csvs:
        print(f'Processing {file_name}')
        df = pd.read_csv(f'./data/{file_name}')
        print(f' - Generating embeddings for {len(df)} documents')
        df['embeddings'] = df['code_snippet'].progress_apply(lambda x: get_document_embedding(x))
        csv_path = f'./data/embeddings_{file_name}'
        df.to_csv(csv_path, index=False)
        print(f' - Embeddings saved to file {csv_path}')
    # Perform semantic search
    semantic_search_predictions = []
    for idx,row in swebench_df.iterrows():
        repo = row["repo"]
        hashes = ast.literal_eval(row["hashes"])
        embeddings_df = pd.read_csv(get_embeddings_file_path(repo))
        associated_rows = embeddings_df[embeddings_df['hash'].isin(hashes)]
        embeddings = associated_rows['embeddings'].apply(ast.literal_eval)
        embeddings = embeddings.apply(lambda x: [float(i) for i in x])
        embeddings = np.array(embeddings.tolist(),  dtype='float32')
        file_paths = associated_rows['file_path'].tolist()
        query_embedding = get_document_embedding(f"{row['problem_statement']} {row['hints_text']}")
        query_embedding = np.expand_dims(query_embedding, axis=0)
        dist, indices = search_top_k_docs(embeddings, query_embedding, 15)
        res_files = [file_paths[i] for i in indices]
        semantic_search_predictions.append(res_files)
        print(f"Processed row {idx}")
    # Write results to csv
    new_df = swebench_df[['modified_files']].copy()
    new_df['predicted_files'] = semantic_search_predictions
    new_df.to_csv('./data/semantic_search_res.csv', index=False)
    # Evaluate tf-idf top k results for each row of the swebench df
    tf_idf_predictions = []
    for idx,row in swebench_df.iterrows():
        repo = row["repo"]
        hashes = ast.literal_eval(row["hashes"])
        embeddings_df = pd.read_csv(get_embeddings_file_path(repo))
        associated_rows = embeddings_df[embeddings_df['hash'].isin(hashes)]
        code_snippets = associated_rows['code_snippet'].to_list()
        file_paths = associated_rows['file_path'].tolist()
        tfidf_matrix, vectorizer = vectorize_documents_tfidf(code_snippets)
        query = f"{row['problem_statement']} {row['hints_text']}"
        res_files = search_tfidf(query, vectorizer, tfidf_matrix, file_paths, 15)
        tf_idf_predictions.append(res_files)
        print(f"Processed row {idx}")
    new_df = swebench_df[['modified_files']].copy()
    new_df['predicted_files'] = semantic_search_predictions
    new_df.to_csv('./data/tfidf_res.csv', index=False)
    
    


