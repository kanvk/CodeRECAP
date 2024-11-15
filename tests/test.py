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
