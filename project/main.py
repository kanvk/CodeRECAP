import streamlit as st
from git import Repo
import os
import csv
import requests
from identifyFunctions import get_function_info
from vectorizeCode import * 
from storeFunctions import *



repo_python_files = []
repo_python_file_names = []

def update_streamlit_output_log(msg, append=True):
    if append:
        st.session_state.log +="  \n"+msg
    else:
        st.session_state.log = msg

def hello_world(name):
    return f"Hello, {name}!"

def is_github_url_valid(repo_url):
    # Return True/False
    # A valid URL must start with https://github.com/
    if not repo_url.startswith("https://github.com/"):
        update_streamlit_output_log("Invalid URL format. It should start with 'https://github.com/'.")
        return False
    # Check the URL by making an HTTP request
    try:
        response = requests.get(repo_url)
        if response.status_code == 200:
            update_streamlit_output_log(f"The repository URL {repo_url} is valid.")
            return True
        elif response.status_code == 404:
            update_streamlit_output_log(f"The repository URL {repo_url} is not found (404).")
            return False 
        else:
            update_streamlit_output_log(f"Received unexpected status code {response.status_code}. The repository URL might be invalid.")
            return False
    except requests.exceptions.RequestException as e:
        update_streamlit_output_log(f"An error occurred: {e}")
        return False

def clone_repository(repo_url):
    # Clone the repo with url repo_url into the cloned_repos directory
    # If repo exists, run git pull instead to update it
    repo_name = repo_url.split("/")[-1]
    repo_name = repo_name[:-4] if repo_name.endswith(".git") else repo_name
    clone_dir = f"./cloned_repos/{repo_name}"
    try:
        if os.path.exists(clone_dir):
            # If the directory exists, perform git pull
            update_streamlit_output_log(f"Directory {clone_dir} already exists. Pulling latest changes...")
            repo = Repo(clone_dir)
            repo.remotes.origin.pull()
            update_streamlit_output_log("Repository updated successfully.")
        else:
            Repo.clone_from(repo_url, clone_dir)
            update_streamlit_output_log(f"Repository cloned successfully into {clone_dir}")
    except Exception as e:
        update_streamlit_output_log(f"An error occurred while cloning the repository: {e}")
        raise e

def analyze_python_files(clone_dir):
    """Analyze all Python files in the cloned repository, including all subdirectories."""
    function_infos = []
    for root, _, files in os.walk(clone_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    code = f.read().strip()
                    # Add all python files into a global list
                    if code:
                        repo_python_files.append(code)
                        repo_python_file_names.append(file)
                    # Debugging: Print the file name and content
                    print(f"Analyzing file: {file_path}")
                    print(f"Content:\n{code}\n")  # Print the content of the file

                    function_info = get_function_info(code)
                    for info in function_info:
                        # Add the file path to each function's information
                        info['file_path'] = file_path
                    function_infos.extend(function_info)  # Add found function info to the list
    return function_infos

def entrypoint(repo_url, query):
    """
    This is the entrypoint function that will be invoked once a URL is input into the streamlit frontend
    """
    # Check if it is a valid GitHub URL
    if not is_github_url_valid(repo_url):
        return
    
    # Extract the repository name from the URL
    repo_name = repo_url.split('/')[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
        
    # Clone the repository into the cloned_repos directory
    try:
        clone_repository(repo_url)
    except Exception as e:
        update_streamlit_output_log(f"Error while cloning the repository: {e}")
        return
    
    # Create a database with the repo_name
    create_database(db_name=repo_name)
    
    # Analyze Python files and get function info
    clone_dir = f"./cloned_repos/{repo_name}"
    function_infos = analyze_python_files(clone_dir)

    if function_infos:
        # Insert function data into the database
        insert_function_data(function_infos, db_name=repo_name)
        for info in function_infos:
            update_streamlit_output_log(
                f"File: {info['file_path']}\n"
                f"Function: {info['name']}, Line Start: {info['start_line']}, "
                f"Line End: {info['end_line']}, Args: {info['arguments']}, "
                f"Varlen Args: {info['varlen_arguments']}, Keyword Args: {info['keyword_arguments']}, "
                f"Positional Args: {info['positional_arguments']}, Varlen Keyword Args: {info['varlen_keyword_arguments']}"
            )
    else:
        update_streamlit_output_log("No Python functions found in the repository.")

    # Initialize the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = RobertaModel.from_pretrained("microsoft/unixcoder-base")

    code_vectors = []
    for code_file in repo_python_files:
        code_vectors.append(vectorize_code(code_file, tokenizer, model))

    csv_file = "embeddings.csv"

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for vec in code_vectors:
            writer.writerow(vec)  # Write the vector directly

    embeddings_query = vectorize_code(query, tokenizer, model)

    matches = get_matches(embeddings_query,code_vectors,3)

    for i, match in enumerate(matches):
        index = match[0]
        update_streamlit_output_log(f"Match {i + 1}: {repo_python_file_names[index-1]}")




def main():
    """
    Main function for the Streamlit app.
    """
    st.set_page_config(page_title="CodeRECAP", layout="wide")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("CodeRECAP")
    
    if 'log' not in st.session_state:
        st.session_state.log = ""  # Initialize log if not already done

    url = st.text_input("Enter the URL to the git repo to be analyzed:")

    query = st.text_input("Enter your query:")

    if url and query:
        entrypoint(url, query)
        st.markdown(st.session_state.log)


if __name__ == "__main__":
    main()

# Example usage: Input a valid URL in the text box. Eg: "https://github.com/kanvk/CodeRECAP.git"
