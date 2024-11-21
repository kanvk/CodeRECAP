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

