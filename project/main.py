import streamlit as st
from git import Repo
import os
import csv
import requests
from identifyFunctions import get_function_info
from llmPrompt import *
from vectorizeCode import * 
from storeFunctions import *

repo_python_files = []
repo_python_file_names = []
# Set up LLaMA client
client, model_name = setup_llama_client()

def update_streamlit_output_log(msg, append=True):
    if append:
        st.session_state.log += "  \n" + msg
    else:
        st.session_state.log = msg

def hello_world(name):
    return f"Hello, {name}!"

def is_github_url_valid(repo_url):
    if not repo_url.startswith("https://github.com/"):
        update_streamlit_output_log("Invalid URL format. It should start with 'https://github.com/'.")
        return False
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
    repo_name = repo_url.split("/")[-1]
    repo_name = repo_name[:-4] if repo_name.endswith(".git") else repo_name
    clone_dir = f"./cloned_repos/{repo_name}"
    try:
        if os.path.exists(clone_dir):
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
    function_infos = []
    for root, _, files in os.walk(clone_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    code = f.read().strip()
                    if code:
                        repo_python_files.append(code)
                        repo_python_file_names.append(file)
                    print(f"Analyzing file: {file_path}")
                    print(f"Content:\n{code}\n")
                    function_info = get_function_info(code)
                    for info in function_info:
                        info['file_path'] = file_path
                    function_infos.extend(function_info)
    return function_infos

def entrypoint(repo_url, query):
    if not is_github_url_valid(repo_url):
        return
    
    repo_name = repo_url.split('/')[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
        
    try:
        clone_repository(repo_url)
    except Exception as e:
        update_streamlit_output_log(f"Error while cloning the repository: {e}")
        return
    
    create_database(db_name=repo_name)
    
    clone_dir = f"./cloned_repos/{repo_name}"
    function_infos = analyze_python_files(clone_dir)

    if function_infos:
        insert_function_data(function_infos, db_name=repo_name)
        repo_summary = "\n".join(
            [
                f"File: {info['file_path']}, Function: {info['name']}, "
                f"Arguments: {info['arguments']}, Start Line: {info['start_line']}, End Line: {info['end_line']}"
                for info in function_infos
            ]
        )

        st.session_state['repo_info'] = repo_summary
    else:
        update_streamlit_output_log("No Python functions found in the repository.")
        st.session_state['repo_info'] = "No functions found in the repository."

    readme_summary = summarize_readme(clone_dir, client, model_name)
    update_streamlit_output_log(f"README Summary:\n{readme_summary}")
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = RobertaModel.from_pretrained("microsoft/unixcoder-base")

    code_vectors_function_level = [vectorize_function(function_info, tokenizer, model) for function_info in function_infos]
    code_vectors = [vectorize_code(code_file, tokenizer, model) for code_file in repo_python_files]

    csv_file = "embeddings.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for vec in code_vectors:
            writer.writerow(vec)

    embeddings_query = vectorize_code(query, tokenizer, model)
    matches = get_matches(embeddings_query, code_vectors, 3)
    matches_function_level = get_function_matches(embeddings_query, code_vectors_function_level, 3)

    output_text = "Top function-level matches:\n"
    for i, (func_name, path, similarity) in enumerate(matches_function_level, start=1):
        output_text += f"{i}. Function: {func_name}\n"
        output_text += f"   Path: {path}\n"
        output_text += f"   Similarity Score: {similarity:.4f}\n\n"

    output_text += "Top file-level matches:\n\n"
    for i, match in enumerate(matches):
        index = match[0]
        if index > 0:
            file_name = repo_python_file_names[index - 1]
            output_text += f"{i + len(matches_function_level)}. File: {file_name}\n"
            output_text += f"   Path: {file_name}\n"

    update_streamlit_output_log(output_text)

def on_submit_question():
    question = st.session_state.get('user_question', '').strip()
    if not question:
        st.write("Please enter a valid question.")
    else:
        repo_info = st.session_state.get('repo_info', 'No repository information available.')
        prompt = (
            f"You are an AI assistant who has analyzed the following Python repository:\n\n"
            f"{repo_info}\n\n"
            f"The user has the following question about the repository:\n"
            f"{question}"
        )

        response = get_response(client, model_name, prompt)
        st.session_state['chat_history'].append(f"User: {question}\nAssistant: {response}")
        st.write(f"Assistant Response: {response}")

def main():
    """
    Main function for the Streamlit app.
    """
    st.set_page_config(page_title="CodeRECAP", layout="wide")  # Set page config first
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
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat_history as an empty list

    url = st.text_input("Enter the URL to the git repo to be analyzed:")
    query = st.text_input("Enter your query:")

    if st.button("Analyze Repository"):
        if url and query:
            entrypoint(url, query)
            st.markdown(st.session_state.log)
        else:
            st.warning("Please enter both a valid repository URL and a query.")

    st.markdown("<hr>", unsafe_allow_html=True)  # Horizontal line for separation
    st.markdown("### Ask Questions About Cloned Repository")  # Title for questions section
    user_question = st.text_input("Enter your question:", key='user_question')

    if st.button("Get Response for Question"):
        on_submit_question()

    if st.session_state.chat_history:
        st.markdown("### Chat History:")  # Title for chat history
        for chat in st.session_state.chat_history:
            st.markdown(chat)

if __name__ == "__main__":
    main()

# Example usage: Input a valid URL in the text box. Eg: "https://github.com/kanvk/CodeRECAP.git"
