# tests/test.py

import pytest

from datasets import load_dataset
import pandas as pd
import sys
import os

# Add the project directory to sys.path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../project'))
sys.path.insert(0, project_path)


from indexingUtils import is_github_url_valid, clone_repository, index_repo_files_and_functions, reset_indexing_output_log
from queryUtils import display_top_k_similar_docs, reset_querying_output_log, display_llm_response

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
                modified_file = parts[2].replace("a/", "", 1)
                modified_files.append(modified_file)
    return modified_files

def get_swebench_dataset() -> pd.DataFrame:
    """Loads the SWE-bench dataset."""
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")
    df = pd.DataFrame(ds["test"])
    df["modified_files"] = df["patch"].apply(get_modified_files)
    df = df[["repo", "base_commit", "modified_files", "problem_statement", "hints_text"]]
    return df

def get_testing_results():
    swebench_df = get_swebench_dataset()
    grouped = swebench_df.groupby('repo')
    
    for repo_name, group_data in grouped:
        print(f"Processing repo: {repo_name}")
        modified_files = group_data['modified_files'].apply(
            lambda files: [os.path.basename(file) for file in files] if isinstance(files, list) else files
        )
        
        # Additional processing or print statements as needed

# def test_hello_world():
#     """Test the hello_world function."""
#     result = hello_world("World")
#     assert result == "Hello, World!"  # Check if the function returns the expected output
# test.py

if __name__ == "__main__":
    get_testing_results()
