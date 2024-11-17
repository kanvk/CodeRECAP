# tests/test.py

import pytest
#from project.main import hello_world  # Adjust the import path as necessary


import pytest
from datasets import load_dataset
import pandas as pd
import sys
import os

# # Add the project directory to sys.path
# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../project'))
# sys.path.insert(0, project_path)

from project.indexingUtils import * #is_github_url_valid, clone_repository, index_repo_files_and_functions, reset_indexing_output_log
from project.queryUtils import * #display_top_k_similar_docs, reset_querying_output_log, display_llm_response


def get_modified_files(patch: str):
    """Gets the list of modified files from a patch."""
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


# Define Pytest-compatible tests
def test_swebench_dataset_loading():
    """Test if SWE-bench dataset loads correctly."""
    swebench_df = get_swebench_dataset()
    assert not swebench_df.empty, "SWE-bench dataset is empty"
    assert "repo" in swebench_df.columns, "Dataset is missing 'repo' column"
    assert "modified_files" in swebench_df.columns, "Dataset is missing 'modified_files' column"


def test_get_modified_files():
    """Test parsing of modified files from a patch."""
    sample_patch = """diff --git a/file1.py b/file1.py
index 123abc..456def 100644
--- a/file1.py
+++ b/file1.py"""
    modified_files = get_modified_files(sample_patch)
    assert modified_files == ["file1.py"], f"Expected ['file1.py'], got {modified_files}"


def test_grouping_by_repo():
    """Test if data can be grouped by repo without issues."""
    swebench_df = get_swebench_dataset()
    grouped = swebench_df.groupby('repo')
    assert len(grouped) > 0, "No repositories found in grouping"
    for repo_name, group_data in grouped:
        assert "modified_files" in group_data.columns, f"'modified_files' missing in group for {repo_name}"


def test_hello_world():
    """Test the hello_world function."""
    result = hello_world("World")
    assert result == "Hello, World!"  # Check if the function returns the expected output

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from datasets import load_dataset
from project.queryUtils import *  
from project.swebenchUtils import sweBenchCloneAndQuery  

def get_modified_files(patch: str):
    # Function to extract modified files from the patch
    modified_files = []
    for line in patch.splitlines():
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) > 2:
                modified_file = parts[2].replace('a/', '', 1)
                modified_files.append(modified_file)
    return modified_files

def get_swebench_dataset() -> pd.DataFrame:
    ds = load_dataset('princeton-nlp/SWE-bench_Verified')
    df = pd.DataFrame(ds['test'])
    df['modified_files'] = df['patch'].apply(get_modified_files)
    #print(df)
    df = df[['repo', 'base_commit', 'modified_files', 'problem_statement', 'hints_text']]  # Assuming there's a ground truth column
    return df

def run_testing_on_all_repos():
    swebench_df = get_swebench_dataset()  # Get the dataset for all repos
    swebench_df['prediction'] = None  # Initialize column for predictions

    all_true_labels = []  # To store ground truth labels (but we don't have these)
    all_predicted_labels = []  # To store predicted labels

    for idx, row in swebench_df.iterrows():
        repo_url = f'https://github.com/{row["repo"]}'
        results = sweBenchCloneAndQuery(repo_url, row['base_commit'], row['problem_statement'], row['hints_text'])

        modified_files = [os.path.basename(file) for file in row['modified_files']] if isinstance(row['modified_files'], list) else row['modified_files']
        
        prediction = 'False'
        for result in results:
            if result in modified_files:
                prediction = 'True'
                break
        swebench_df.loc[idx, 'prediction'] = prediction

        # Append prediction for later inspection
        all_predicted_labels.append(prediction == 'True')
        
        # Print the repo, true/false, and prediction for manual review
        print(f"Repo: {row['repo']}, Predicted: {prediction}")

    # Print predictions manually
    print(f"Predictions: {all_predicted_labels}")

# Run testing on all repos
run_testing_on_all_repos()

