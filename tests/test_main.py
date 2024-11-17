from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from datasets import load_dataset
from project.queryUtils import sweBenchCloneAndQuery  # Assuming this function exists

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
    df = df[['repo', 'base_commit', 'modified_files', 'problem_statement', 'hints_text', 'ground_truth_label']]  # Assuming there's a ground truth column
    return df

def run_testing_on_all_patches():
    swebench_df = get_swebench_dataset()
    swebench_df['prediction'] = None  # Initialize column for predictions

    all_true_labels = []  # To store ground truth labels
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

        # Append ground truth and prediction for evaluation later
        all_true_labels.append(row['ground_truth_label'])
        all_predicted_labels.append(prediction == 'True')

    # Evaluate model performance
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    precision = precision_score(all_true_labels, all_predicted_labels)
    recall = recall_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Save predictions and evaluation results
    swebench_df.to_csv('swebench_predictions.csv', index=False)

    return swebench_df

# If you want to call this directly when the script runs:
if __name__ == "__main__":
    run_testing_on_all_patches()
