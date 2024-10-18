import streamlit as st
from git import Repo
import os
import requests

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

def entrypoint(repo_url):
    """
    This is the entrypoint function that will be invoked once a URL is input into the streamlit frontend
    """
    # Check if it is a valid github url
    if not is_github_url_valid(repo_url):
        return
    # Clone the repository into the cloned_repos directory
    clone_repository(repo_url)
    # TODO: Index all python files from the cloned repository and store it in the database


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
    url = st.text_input("Enter the URL to the git repo to be analysed:")
    st.session_state.log = ""
    if url:
        entrypoint(url)
        st.markdown(st.session_state.log)



if __name__ == "__main__":
    main()

# Example usage: Input a valid URL in the text box. Eg: "https://github.com/kanvk/CodeRECAP.git"
