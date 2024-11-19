import streamlit as st
import matplotlib.pyplot as plt
import os
from indexingUtils import (
    is_github_url_valid,
    clone_repository,
    index_repo_files_and_functions,
    reset_indexing_output_log,
)
import indexingUtils
from queryUtils import (
    display_top_k_similar_docs,
    reset_querying_output_log,
    display_llm_response,
    display_top_k_similar_docs_tfidf,
)

from swebenchUtils import (
    locate_files,
    sweBenchCloneAndQuery
)

from benchmark import (
    get_swebench_dataset
)

# Example usage: Input a valid URL in the text box. Eg: "https://github.com/kanvk/CodeRECAP.git"


def hello_world(name):
    return f"Hello, {name}!"


def index_repo(url, commit_hash=None):
    reset_indexing_output_log()
    # Check if url is valid
    if not is_github_url_valid(url):
        return
    # Clone the repo if its valid
    try:
        repo_name, cloned_dir = clone_repository(repo_url=url, commit_hash=commit_hash)
        st.session_state.repo_name = repo_name
    except Exception:
        return
    # index files and functions
    index_repo_files_and_functions(repo_name, cloned_dir)


def query_repo(query_text, k = 5, display = True):
    reset_querying_output_log()
    # Vector similarity for functions
    #function_results = display_top_k_similar_docs(
    #    st.session_state.functions_vector_store, query_text, k, "function", display
    #)
    # Vector similarity for files
    file_results = display_top_k_similar_docs(
        st.session_state.files_vector_store, query_text, k, "file", display
    )
    # TF-IDF similarity for files
    tfidf_results = display_top_k_similar_docs_tfidf(
        query_text, k, "file", display
    )
    # LLM Query
    display_llm_response(query_text)
    
    return file_results, tfidf_results


def main():
    """
    Main function for the Streamlit app.
    """
    # Webpage Header Section
    st.set_page_config(page_title="CodeRECAP", layout="wide")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("CodeRECAP")

    # Initialize logs for the index repository section and query repository section
    if "indexing_log" not in st.session_state:
        st.session_state.indexing_log = ""
    if "querying_log" not in st.session_state:
        st.session_state.querying_log = ""

    # Index Repo Section
    st.header("Index Repository")
    repo_url = st.text_input("Enter the URL to the git repo to be analyzed")
    # Trigger index_repo function when button is clicked or Enter is pressed
    if st.button("Index"):
        if repo_url:
            index_repo(repo_url)
    st.write(st.session_state.indexing_log)

    # Query Repo Section
    st.header("Query Repository")
    query_text = st.text_input("Enter your Query")
    # Trigger query_repo function when button is clicked or Enter is pressed
    if st.button("Query"):
        if query_text:
            query_repo(query_text)
    st.write(st.session_state.querying_log)

    # Testing
    st.header("Perform Testing")
    if st.button("Test"):
        swebench_df = get_swebench_dataset()
        swebench_df['prediction'] = None

        # Group the DataFrame by 'repo'
        grouped = swebench_df.groupby('repo')
        for repo_name, group_data in grouped:
            for idx, row in group_data.iterrows():
                print(f"Processing row: {idx} for repo: {repo_name}")
                repo_url = f"https://github.com/{row['repo']}"
                results = sweBenchCloneAndQuery(repo_url, row['base_commit'], row['problem_statement'], row['hints_text'])
                # Dummy results for testing purposes
                # results = ['separable.py', 'sympy_parser.py']

                # Process the modified_files for each row
                modified_files = [os.path.basename(file) for file in row['modified_files']] if isinstance(row['modified_files'], list) else row['modified_files']
                
                # Check if the prediction is in modified_files
                prediction = 'False'  # Default to 'False'
                for result in results:
                    if result in modified_files:
                        prediction = 'True'
                        break
                
                swebench_df.loc[idx, 'prediction'] = prediction
                
        # After processing, store the DataFrame to a CSV file
        print("Saving predictions to CSV file... (swebench_predictions.csv)\n")
        swebench_df.to_csv('swebench_predictions.csv', index=False)
        
        # clear log and display message
        reset_querying_output_log()
        
        st.session_state.testing_log = ""
        # Convert the 'prediction' column from string to boolean
        swebench_df['prediction'] = swebench_df['prediction'].apply(lambda x: x == 'True')

        # Calculate the count of True and False predictions
        true_count = swebench_df['prediction'].sum()  # Sum of True values
        false_count = len(swebench_df) - true_count   # Count of False values
        total_predictions = len(swebench_df)

        # Calculate the percentage of True and False predictions
        true_percentage = (true_count / total_predictions) * 100
        false_percentage = (false_count / total_predictions) * 100

        # Display the proportions in Streamlit
        st.session_state.testing_log += f"**Proportion of True Predictions:** {true_percentage:.2f}% {'\u00A0' * 15}"
        st.session_state.testing_log += f"**Proportion of False Predictions:** {false_percentage:.2f}%\n\n"
        st.session_state.testing_log += f"**Number of True Predictions:** {true_count} {'\u00A0' * 15}"
        st.session_state.testing_log += f"**Number of False Predictions:** {false_count}\n\n"
        st.write(st.session_state.testing_log)
                
        # Display a pie chart to visualize the proportions
        fig, ax = plt.subplots()
        ax.pie([true_percentage, false_percentage], labels=['True', 'False'], autopct='%1.1f%%', colors=['#4CAF50', '#FF6347'])
        ax.set_title('Proportion of True vs False Predictions')
        st.pyplot(fig)

if __name__ == "__main__":
    indexingUtils.streamlit_log = True
    main()

# Example usage: Input a valid URL in the text box, optionally followed by a branch or a hash. Eg: "https://github.com/kanvk/CodeRECAP.git branch:main"
