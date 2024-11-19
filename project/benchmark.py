from datasets import load_dataset
import pandas as pd


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
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")
    df = pd.DataFrame(ds["test"])
    df["modified_files"] = df["patch"].apply(get_modified_files)
    df = df[
        ["repo", "base_commit", "modified_files", "problem_statement", "hints_text"]
    ]
    return df
