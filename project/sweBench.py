import os
import pandas as pd
from langsmith import Client
from datasets import load_dataset
from langsmith.evaluation import evaluate
from indexingUtils import is_github_url_valid, clone_repository, index_repo_files_and_functions, reset_indexing_output_log
from queryUtils import display_top_k_similar_docs, reset_querying_output_log

# Ensure the LangSmith API key is set in the environment
api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    raise ValueError("LANGCHAIN_API_KEY environment variable not set")

client = Client(api_key=api_key)

# Function to load and upload SWE-Bench dataset, set test repositories, and evaluate it
def swebench_evaluate():
    try:
        # Load the SWE-Bench dataset from Hugging Face
        print("Loading SWE-Bench dataset...")
        dataset = load_dataset("princeton-nlp/SWE-bench", split="dev")
        
        print(f"Dataset loaded with {len(dataset)} examples.")
        # Convert the dataset to a Pandas DataFrame
        df = pd.DataFrame(dataset)

        print("Dataset loaded successfully. Processing the data...")
        # Print the first few rows to inspect the available fields
        print("Inspecting dataset fields...")
        print(df.head())  # Check the first few rows to confirm column names
        
        # Apply version formatting
        df['version'] = df['version'].apply(lambda x: f"version:{x}")
        
        # Save to CSV
        csv_file_path = "./SWE-bench.csv"
        df.to_csv(csv_file_path, index=False)

        print(f"Dataset saved to CSV: {csv_file_path}")

        # Upload CSV to LangSmith
        print("Uploading dataset to LangSmith...")
        uploaded_dataset = client.upload_csv(
            csv_file=csv_file_path,
            input_keys=list(df.columns),  # Dynamically extract input columns
            output_keys=[],  # Adjust this if needed
            name="swe-bench-programatic-upload",
            description="SWE-bench dataset",
            data_type="kv"
        )
        
        # Debug: Print the structure of uploaded_dataset
        print(f"Uploaded dataset structure: {uploaded_dataset}")

        # Check if upload was successful
        if isinstance(uploaded_dataset, dict):
            dataset_id = uploaded_dataset.get('id')
            print(f"Dataset uploaded successfully. Dataset ID: {dataset_id}")
        else:
            print("Uploaded dataset is not a dictionary-like object. Raw output:", uploaded_dataset)

        # Ask for the dataset ID from the user
        dataset_id = input("Enter the dataset ID: ")  # Input the dataset ID

        # Retrieve the first 3 repositories to be used as test cases
        test_repos = dataset.select([0, 1, 2])  # Selecting the first 3 test cases
        test_repo_urls = [test_repos[i].get('repo') for i in range(3)]  # Get repo URLs from selected dataset
        test_instance_ids = [test_repos[i].get('instance_id') for i in range(3)]  # Get instance IDs
        
        print(f"Test Repositories Selected: {test_repo_urls}")

        # Create a list of test repositories with their instance IDs
        test_repositories = [{"repo_url": repo_url, "instance_id": instance_id} 
                             for repo_url, instance_id in zip(test_repo_urls, test_instance_ids)]
        
        print(f"Test Repositories: {test_repositories}")

        # Define a predict function that uses your agent
        def predict(inputs: dict):
            print(f"Predict called with inputs: {inputs}")
            repo_url = inputs.get("repo_url")  # Now accessing 'repo_url'
            # if not repo_url or not is_github_url_valid(repo_url):
            #     return {"error": "Invalid repository URL"}
            
            try:
                print(f"Cloning repository: {repo_url}")
                # Index the repository
                reset_indexing_output_log()
                repo_name, cloned_dir = clone_repository(repo_url=repo_url)
                print(f"Repository cloned: {repo_name}, Directory: {cloned_dir}")
                index_repo_files_and_functions(repo_name, cloned_dir)
                print("Repository functions indexed.")

                # Use the instance ID or a relevant field from SWE-Bench as the query
                query_text = f"Summarize the functions for SWE-Bench evaluation on {inputs['instance_id']}"
                print(f"Query generated: {query_text}")

                # Reset query logs and get similar documents
                reset_querying_output_log()
                function_similarities = display_top_k_similar_docs(
                    st.session_state.functions_vector_store, query_text, 5, "function"
                )
                print(f"Function similarities found: {function_similarities}")

                # Return the results in a format compatible with SWE-Bench evaluation
                return {
                    "instance_id": inputs["instance_id"],
                    "model_patch": function_similarities,  # Mocked patch details
                    "model_name_or_path": repo_name
                }
            except Exception as e:
                print(f"Error in prediction: {e}")
                return {"error": f"An error occurred during prediction: {str(e)}"}

        # Run evaluation on the selected test repositories
        print("Running evaluation on test repositories...")
        result = evaluate(
            predict,
            data=client.list_examples(dataset_id=dataset_id, splits=["test"]),
        )
        print("Evaluation complete.")
        
        return result

    except Exception as e:
        print(f"Error in SWE-Bench evaluation: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

# Call the evaluation function
print("Starting SWE-Bench evaluation...")
evaluation_results = swebench_evaluate()

# Print or log the evaluation results
print("Evaluation Results:", evaluation_results)
