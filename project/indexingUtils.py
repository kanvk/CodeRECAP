import os
import requests
from git import Repo
import streamlit as st
from project.identifyFunctions import get_function_info
from project.storeFunctions import create_database, insert_function_data
from project.vectorizeCode import (
    create_documents_from_code_infos,
    save_or_update_faiss_vector_store,
    UniXcoderEmbeddings,
)
from project.tfidf import vectorize_documents_tfidf

streamlit_log = False

def update_indexing_output_log(msg, append=True):
    if streamlit_log:
        if append:
            st.session_state["indexing_log"] += "  \n" + msg
        else:
            st.session_state["indexing_log"] = msg
    print(msg)


def reset_indexing_output_log():
    st.session_state["indexing_log"] = ""


def is_github_url_valid(repo_url):
    # Return True/False
    # A valid URL must start with https://github.com/
    if not repo_url.startswith("https://github.com/"):
        update_indexing_output_log(
            "Invalid URL format. It should start with 'https://github.com/'."
        )
        return False
    # Check the URL by making an HTTP request
    try:
        response = requests.get(repo_url)
        if response.status_code == 200:
            update_indexing_output_log(
                f"The repository URL {repo_url} is valid.")
            return True
        elif response.status_code == 404:
            update_indexing_output_log(
                f"The repository URL {repo_url} is not found (404)."
            )
            return False
        else:
            update_indexing_output_log(
                f"Received unexpected status code {response.status_code}. The repository URL might be invalid."
            )
            return False
    except requests.exceptions.RequestException as e:
        update_indexing_output_log(f"An error occurred: {e}")
        return False


def clone_repository(repo_url, branch=None, commit_hash=None):
    # Clone the repo with url=repo_url into the cloned_repos directory
    # If repo already exists under cloned_repos, run git pull instead to update it
    repo_name = repo_url.split("/")[-1]
    repo_name = repo_name[:-4] if repo_name.endswith(".git") else repo_name
    clone_dir = f"./cloned_repos/{repo_name}"
    try:
        # If the directory exists, perform git pull, else clone it
        if os.path.exists(clone_dir):
            update_indexing_output_log(
                f"Directory {clone_dir} already exists. Pulling latest changes."
            )
            repo = Repo(clone_dir)
            repo.git.checkout("main") # if it was previously checked out to a random commit, switch to main and pull to update
            repo.remotes.origin.pull()
            update_indexing_output_log("Repository updated successfully.")
        else:
            Repo.clone_from(repo_url, clone_dir)
            update_indexing_output_log(
                f"Repository cloned successfully into {clone_dir}"
            )
        # if a branch is passed as arguments to the method, checkout to that branch
        if branch:
            repo = Repo(clone_dir)
            repo.git.checkout(branch)
            update_indexing_output_log(
                f"Checked out to branch {branch} in {clone_dir}")
        # if a commit_hash is passed as arguments to the method, checkout to that commit_hash
        if commit_hash:
            repo = Repo(clone_dir)
            repo.git.checkout(commit_hash)
            update_indexing_output_log(
                f"Checked out to commit {commit_hash} in {clone_dir}"
            )
        return repo_name, clone_dir
    except Exception as e:
        update_indexing_output_log(
            f"An error occurred while cloning the repository: {e}"
        )
        raise e


def analyze_python_files(clone_dir):
    """Analyze all Python files in the cloned repository, including all subdirectories. Identify all functions found in the files and their metadata (start line, end line, args, etc)"""
    function_infos = []
    file_infos = []
    for root, _, files in os.walk(clone_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    code = f.read().strip()
                    # Add all python files into a global list
                    if code:
                        file_infos.append(
                            {"file_path": file_path, "name": file,
                                "code_snippet": code}
                        )
                    function_info = get_function_info(code)
                    for info in function_info:
                        # Add the file path to each function's information
                        info["file_path"] = file_path
                    # Add found function info to the list
                    function_infos.extend(function_info)
    files_list = [f["file_path"] for f in file_infos]
    return file_infos, files_list, function_infos


def index_repo_files_and_functions(repo_name, clone_dir):
    # Create a database with the repo_name
    create_database(db_name=repo_name)
    update_indexing_output_log(
        f"Indexing repository: {repo_name} found at {clone_dir}")
    # Analyze Python files and get function details like start line no, end line no, arguments, etc
    file_infos, files_list, function_infos = analyze_python_files(clone_dir)
    st.session_state.file_infos = file_infos
    st.session_state.function_infos = function_infos
    st.session_state.files_list = files_list
    # Insert function data into the database
    if function_infos:
        insert_function_data(function_infos, db_name=repo_name)
        update_indexing_output_log(
            f"Identified {len(function_infos)} functions")
        for info in function_infos:
            # Log details to streamlit ui
            update_indexing_output_log(
                f"File: {info['file_path']}, Function: {info['name']}, Line Start: {info['start_line']}, "
                f"Line End: {info['end_line']}, Args: {info['arguments']}, "
                f"Varlen Args: {info['varlen_arguments']}, Keyword Args: {info['keyword_arguments']}, "
                f"Positional Args: {info['positional_arguments']}, Varlen Keyword Args: {info['varlen_keyword_arguments']}"
            )
    else:
        update_indexing_output_log(
            "No Python functions found in the repository.")
    # Vectorize functions and files and save in vector stores
    unixcoder_embedding_model = UniXcoderEmbeddings(
        model_name="microsoft/unixcoder-base"
    )
    function_docs = create_documents_from_code_infos(
        st.session_state.function_infos)
    file_docs = create_documents_from_code_infos(st.session_state.file_infos)

    st.session_state.tfidf_matrix, st.session_state.tfidf_vectorizer = (
        vectorize_documents_tfidf(st.session_state.file_infos)
    )
    if not os.path.exists(f"vector_stores/{repo_name}"):
        os.makedirs(f"vector_stores/{repo_name}")
    update_indexing_output_log(
        f"Identified {len(function_docs)} functions. Indexing ....."
    )
    if len(function_infos)>1000:
        update_indexing_output_log("Too many functions identified. Skipping vectorization")
    else:
        st.session_state.functions_vector_store = save_or_update_faiss_vector_store(
            f"vector_stores/{repo_name}/functions_index",
            function_docs,
            unixcoder_embedding_model,
        )
    update_indexing_output_log(
        f"Identified {len(file_docs)} python files. Indexing .....")
    st.session_state.files_vector_store = save_or_update_faiss_vector_store(
        f"vector_stores/{repo_name}/files_index", file_docs, unixcoder_embedding_model
    )
