from datasets import load_dataset
from indexingUtils import *
import re

def parse_patch(patch):
    """
    Parses a git-style patch and extracts all changed file paths.
    
    Args:
    patch (str): The patch in unified diff format.
    
    Returns:
    changed_files (list): A list of unique file paths that were modified.
    """
    changed_files = set()  # Use a set to avoid duplicates
    
    # Split the patch into lines
    patch_lines = patch.splitlines()
    
    # Regular expressions to match the lines that indicate file changes
    file_pattern = re.compile(r'^\+\+\+ (.+)$|^--- (.+)$')
    
    for line in patch_lines:
        match = file_pattern.match(line)
        if match:
            # Extract the file path from the matched group
            file_path = match.group(1) or match.group(2)
            
            # Remove 'a/' and 'b/' prefixes from the file path
            file_path = file_path.lstrip('ab/')
            
            # Add the file path to the set (automatically handles uniqueness)
            changed_files.add(file_path)

    return list(changed_files)
def clone_swe_bench():
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")

    print(ds)

    test_data = ds['test']

    # Get the first entry's repo URL and base commit hash
    first_repo_url = test_data['repo'][0]
    first_commit_hash = test_data['base_commit'][0]
    patch = test_data['patch'][0]
    print(patch)
    
    # Parse the patch to get the list of changed files
    changed_files = parse_patch(patch)
    print(changed_files)

# Clone the repository for the first entry
# repo_name, clone_dir = clone_repository_add_github(first_repo_url, commit_hash=first_commit_hash)
# print(repo_name, clone_dir)


clone_swe_bench()
