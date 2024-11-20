import streamlit as st
import matplotlib.pyplot as plt
import os

from queryUtils import (
    display_top_k_similar_docs_tfidf,
    reset_querying_output_log
)

from tfidf import (
    vectorize_documents_tfidf
)

from swebenchUtils import (
    locate_files,
    sweBenchClone,
    sweBenchQuery
)

from benchmark import (
    get_swebench_dataset
)

def test_swebench(top_k_tfidf = 10, rows = 5):
    swebench_df = get_swebench_dataset().head(rows)
    swebench_df['prediction_llm'] = None
    swebench_df['prediction_tfidf'] = None

    # Group the DataFrame by 'repo'
    grouped = swebench_df.groupby('repo')
    for repo_name, group_data in grouped:
        for idx, row in group_data.iterrows():
            print(f"Processing row: {idx} for repo: {repo_name}")
            repo_url = f"https://github.com/{row['repo']}"
            
            # Clone repo and analyze files in it 
            files_list, file_infos, function_infos = sweBenchClone(repo_url, row['base_commit'])
            # Get LLM response
            llm_results = sweBenchQuery(files_list, repo_name, row['problem_statement'], row['hints_text'])

            # Dummy results for testing purposes
            # llm_results = ['separable.py', 'sympy_parser.py']

            # Process the modified_files for each row
            modified_files = [os.path.basename(file) for file in row['modified_files']] if isinstance(row['modified_files'], list) else row['modified_files']
            
            # Check if every file in modified_files exists in results
            prediction_llm = 'True' if all(file in llm_results for file in modified_files) else 'False'
            swebench_df.loc[idx, 'prediction_llm'] = prediction_llm
            
            
            # Do testing for tfidf  
            # construct a query for tfidf based on the problem statement and hints
            query_tf_idf = f"{row['problem_statement']} {row['hints_text']}"
            
            st.session_state.tfidf_matrix, st.session_state.tfidf_vectorizer = (vectorize_documents_tfidf(st.session_state.file_infos))
            tfidf_results = display_top_k_similar_docs_tfidf(query_tf_idf, top_k_tfidf, "file", False)
            
            # Check if every file in modified_files exists in results
            prediction_tfidf = 'True' if all(file in tfidf_results for file in modified_files) else 'False'
            swebench_df.loc[idx, 'prediction_tfidf'] = prediction_tfidf
                

    # After processing, store the DataFrame to a CSV file
    print("Saving predictions to CSV file... (swebench_predictions.csv)\n")
    swebench_df.to_csv('swebench_predictions.csv', index=False)         
    
    return swebench_df  


def display_test_results (swebench_df, col):
        
        # clear log and display message
        reset_querying_output_log()
        
        st.session_state.testing_log = ""
        # Convert the 'prediction' column from string to boolean
        swebench_df[col] = swebench_df[col].apply(lambda x: x == 'True')

        # Calculate the count of True and False predictions
        true_count = swebench_df[col].sum()  # Sum of True values
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
        ax.set_title(f'Proportion of True vs False Predictions : {col}')
        st.pyplot(fig)