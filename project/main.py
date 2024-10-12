import streamlit as st
from git import Repo


def hello_world(name):
    return f"Hello, {name}!"


def clone_repository(repo_url):
    # Clone the repo with url repo_url into the cloned_repos directory
    # TODO: Check if repo exists and run pull instead of clone if it exists
    repo_name = repo_url.split("/")[-1][:-4]
    clone_dir = f"./cloned_repos/{repo_name}"
    try:
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
    name = st.text_input("Enter your name:")
    if name:
        st.markdown(f"Hello, {name}!")


if __name__ == "__main__":
    main()

# Example usage
# repo_url = "https://github.com/kanvk/CodeRECAP.git"
