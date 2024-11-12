import streamlit as st
from indexingUtils import (
    is_github_url_valid,
    clone_repository,
    index_repo_files_and_functions,
    reset_indexing_output_log,
)
from queryUtils import (
    display_top_k_similar_docs,
    reset_querying_output_log,
    display_llm_response,
)

# Example usage: Input a valid URL in the text box. Eg: "https://github.com/kanvk/CodeRECAP.git"


def hello_world(name):
    return f"Hello, {name}!"


def index_repo(url):
    reset_indexing_output_log()
    # Check if url is valid
    if not is_github_url_valid(url):
        return
    # Clone the repo if its valid
    try:
        repo_name, cloned_dir = clone_repository(repo_url=url)
        st.session_state.repo_name = repo_name
    except Exception:
        return
    # index files and functions
    index_repo_files_and_functions(repo_name, cloned_dir)


def query_repo(query_text):
    reset_querying_output_log()
    # Vector similarity for functions
    display_top_k_similar_docs(
        st.session_state.functions_vector_store, query_text, 5, "function"
    )
    # Vector similarity for files
    display_top_k_similar_docs(
        st.session_state.files_vector_store, query_text, 5, "file"
    )
    # LLM Query
    display_llm_response(query_text)


def main():
    """
    Main function for the Streamlit app.
    """
    # Webpage Header Section
    st.set_page_config(page_title="CodeRECAP", layout="wide")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("CodeRECAP")

    # Initialize logs for the index repository section and query repository section
    if "indexing_log" not in st.session_state:
        st.session_state.indexing_log = ""
    if "querying_log" not in st.session_state:
        st.session_state.querying_log = ""

    # Index Repo Section
    st.header("Index Repository")
    repo_url = st.text_input("Enter the URL to the git repo to be analyzed")
    # Trigger index_repo function when button is clicked or Enter is pressed
    if st.button("Index Now"):
        if repo_url:
            index_repo(repo_url)
    st.write(st.session_state.indexing_log)

    # Query Repo Section
    st.header("Query Repository")
    query_text = st.text_input("Enter your Query")
    # Trigger query_repo function when button is clicked or Enter is pressed
    if st.button("Query Now"):
        if query_text:
            query_repo(query_text)
    st.write(st.session_state.querying_log)


if __name__ == "__main__":
    main()

# Example usage: Input a valid URL in the text box, optionally followed by a branch or a hash. Eg: "https://github.com/kanvk/CodeRECAP.git branch:main"
