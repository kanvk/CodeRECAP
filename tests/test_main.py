import os
import pandas as pd
from datasets import load_dataset
from project.indexingUtils import clone_repository, is_github_url_valid
from project.queryUtils import sweBenchCloneAndQuery  # Assuming this function exists

# Assuming other necessary imports and utility functions

def get_modified_files(patch: str):
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
    df = df[['repo', 'base_commit', 'modified_files', 'problem_statement', 'hints_text']]
    return df

def run_testing_on_all_patches():
    swebench_df = get_swebench_dataset()
    swebench_df['prediction'] = None  # Initialize column for predictions

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

    swebench_df.to_csv('swebench_predictions.csv', index=False)
    return swebench_df

# If you want to call this directly when the script runs:
if __name__ == "__main__":
    run_testing_on_all_patches()
