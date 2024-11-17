from datasets import load_dataset
import os
import pandas as pd
import streamlit as st
from project.indexingUtils import is_github_url_valid, clone_repository, index_repo_files_and_functions, reset_indexing_output_log
from project.queryUtils import display_top_k_similar_docs, reset_querying_output_log, display_llm_response

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
    print("in")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")
    df = pd.DataFrame(ds["test"])
    df["modified_files"] = df["patch"].apply(get_modified_files)
    df = df[
        ["repo", "base_commit", "modified_files", "problem_statement", "hints_text"]
    ]
    return df

def get_testing_results():
    swebench_df = get_swebench_dataset()

    # Group the DataFrame by 'repo'
    grouped = swebench_df.groupby('repo')
    for repo_name, group_data in grouped:
        print(group_data['base_commit'])
        for data in group_data:
            print(data)
        print(f"Processing repo: {repo_name}")
        modified_files = group_data['modified_files'].apply(
            lambda files: [os.path.basename(file) for file in files] if isinstance(files, list) else files
        )
    #index_repo('https://github.com/kanvk/CodeRECAP')
    #index_repo('https://github.com/sqlfluff/sqlfluff/commit/a820c139ccbe6d1865d73c4a459945cd69899f8f')
        # Now, you can apply any operations you need to process the data for this specific repo
        # Example: You can apply a function or analyze the data per repo


print(get_testing_results())