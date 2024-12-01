1. Run get_data.py
    - It collects the swebench verified dataset and transforms it as required
    - Clones each repo at the required commit, identifies all files at the commit and get the embeddings of each file at that commit
    - Performs semantic embedding search and identifies the top k results for the problem statement
    - Perfroms tf-idf vectoriaton for all files and identifies the top k results for the problem statement
    - Performs LLM Prompting to identify the top files to be modified to solve the problem statement
    - All these results are written to csv files in the data folder
2. Run evaluate_results.py
    - Gives the precision@K and recall@K scores for all 3 methods - LLM Prompting, Semantic Embedding Search and Tf-Idf Vector Search