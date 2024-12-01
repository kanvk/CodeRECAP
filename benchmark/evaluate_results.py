import ast
import json
import numpy as np
import pandas as pd


def get_metrics_for_instance(ground_truth, pred, k):
    count = 0
    for i in ground_truth:
        if i in pred[:k]:
            count+=1
    precision = count/k if len(ground_truth)>k else count/len(ground_truth)
    recall = count/len(ground_truth)
    return (recall,precision)

def compute_metrics(ground_truth, predictions, k_values):
    recall_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    for gt,pred in zip(ground_truth, predictions):
        for k in k_values:
            recall,precision = get_metrics_for_instance(gt, pred, k)
            recall_scores[k].append(recall)
            precision_scores[k].append(precision)
    results = {}
    for k in recall_scores:
        results[f"recall@{k}"] = np.mean(recall_scores[k])
        results[f"precision@{k}"] = np.mean(precision_scores[k])
    return results

if __name__=="__main__":
    swebench_df = pd.read_csv(f'./data/swebench_updated.csv')
    ground_truth = list(swebench_df['modified_files'].apply(ast.literal_eval))
    # Initialize an empty list to store the JSON objects
    llm_results = []
    # Open the jsonl file for reading
    with open("./data/results.jsonl", 'r') as file:
        for line in file:
            # Parse each line as a JSON object and append it to the list
            llm_results.append(json.loads(line.strip()))
    llm_predictions = []
    for res in llm_results:
        files = []
        if res['answered']:
            for i in range(1,res['num_files']+1):
                files.append(res['files'][f'rank_{i}'])
        llm_predictions.append(files)
    # Evaluate and print LLM Prompting results
    print("Swebench LLM Basic Query Results : ")
    res = compute_metrics(ground_truth, llm_predictions, [1,2,3])
    for key in res:
        print(f"{key} : {res[key]}")
    # Evaluate and print semantic search results
    new_df = pd.read_csv('./data/semantic_search_res.csv')
    ss_predictions = new_df['predicted_files'].apply(ast.literal_eval).to_list()
    print("Swebench Semantic Search Results : ")
    res = compute_metrics(ground_truth, ss_predictions, [1,2,3,5,10,15])
    for key in res:
        print(f"{key} : {res[key]}")
    # Evaluate and print tf-idf results
    new_df = pd.read_csv('./data/tfidf_res.csv')
    tf_idf_predictions = new_df['predicted_files'].apply(ast.literal_eval).to_list()
    print("Swebench Tf-Idf Results : ")
    res = compute_metrics(ground_truth, tf_idf_predictions, [1,2,3,5,10,15])
    for key in res:
        print(f"{key} : {res[key]}")



