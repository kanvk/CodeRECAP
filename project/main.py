import streamlit as st
import os
from indexingUtils import (
    is_github_url_valid,
    clone_repository,
    index_repo_files_and_functions,
    reset_indexing_output_log,
)
import indexingUtils
from queryUtils import (
    display_top_k_similar_docs,
    reset_querying_output_log,
    display_llm_response,
    display_top_k_similar_docs_tfidf,
)

from swebenchUtils import (
    locate_files
)

from benchmark import (
    get_swebench_dataset
)

# Example usage: Input a valid URL in the text box. Eg: "https://github.com/kanvk/CodeRECAP.git"


def hello_world(name):
    return f"Hello, {name}!"


def index_repo(url, commit_hash=None):
    reset_indexing_output_log()
    # Check if url is valid
    if not is_github_url_valid(url):
        return
    # Clone the repo if its valid
    try:
        repo_name, cloned_dir = clone_repository(repo_url=url, commit_hash=commit_hash)
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
    # TF-IDF similarity for files
    display_top_k_similar_docs_tfidf(query_text, 5, "file")
    # LLM Query
    #display_llm_response(query_text)

def run_swe_bench_evaluation():
    try:
        evaluation_results = swebench_evaluate()
        st.session_state.swebench_log = evaluation_results
    except Exception as e:
        st.session_state.swebench_log = f"Error during SWE-Bench evaluation: {str(e)}"


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

    if "swebench_log" not in st.session_state:
        st.session_state.swebench_log = ""
    # Index Repo Section
    st.header("Index Repository")
    repo_url = st.text_input("Enter the URL to the git repo to be analyzed")
    # Trigger index_repo function when button is clicked or Enter is pressed
    if st.button("Index"):
        if repo_url:
            index_repo(repo_url)
    st.write(st.session_state.indexing_log)

    # Query Repo Section
    st.header("Query Repository")
    query_text = st.text_input("Enter your Query")
    # Trigger query_repo function when button is clicked or Enter is pressed
    if st.button("Query"):
        if query_text:
            query_repo(query_text)
    st.write(st.session_state.querying_log)

    # Testing
    st.header("Perform Testing")
    if st.button("Test"):
        swebench_df = get_swebench_dataset()
        swebench_df['prediction'] = None

        # Group the DataFrame by 'repo'
        grouped = swebench_df.groupby('repo')
        for repo_name, group_data in grouped:
            print(f"Processing repo: {repo_name}")
            for idx, row in group_data.iterrows():
                repo_url = f"https://github.com/{row['repo']}"
                results = locate_files(repo_url, row['base_commit'], row['problem_statement'], row.get("hints_text"))

                # Process the modified_files for each row
                modified_files = [os.path.basename(file) for file in row['modified_files']] if isinstance(row['modified_files'], list) else row['modified_files']
                
                # Check if the prediction is in modified_files
                prediction = 'False'  # Default to 'False'
                for result in results:
                    if result in modified_files:
                        prediction = 'True'
                        break
                
                swebench_df.loc[idx, 'prediction'] = prediction
        
        # After processing, store the DataFrame to a CSV file
        swebench_df.to_csv('swebench_predictions.csv', index=False)
        

if __name__ == "__main__":
    indexingUtils.streamlit_log = True
    main()

# Example usage: Input a valid URL in the text box, optionally followed by a branch or a hash. Eg: "https://github.com/kanvk/CodeRECAP.git branch:main"
