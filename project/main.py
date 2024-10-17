import streamlit as st
from git import Repo
import os


def hello_world(name):
    return f"Hello, {name}!"


def clone_repository(repo_url):
    # Clone the repo with url repo_url into the cloned_repos directory
    # If repo exists, run git pull instead to update it
    repo_name = repo_url.split("/")[-1][:-4]
    clone_dir = f"./cloned_repos/{repo_name}"
    try:
        if os.path.exists(clone_dir):
            # If the directory exists, perform git pull
            status = f"Directory {clone_dir} already exists. Pulling latest changes..."
            repo = Repo(clone_dir)
            repo.remotes.origin.pull()
            status+=" Repository updated successfully."
        else:
            Repo.clone_from(repo_url, clone_dir)
            status = f"Repository cloned successfully into {clone_dir}"
    except Exception as e:
        status = f"An error occurred while cloning the repository: {e}"
    return status


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
    if url:
        st.markdown(f"{clone_repository(url)}")



if __name__ == "__main__":
    main()

# Example usage: Input this URL in the text box: "https://github.com/kanvk/CodeRECAP.git"
