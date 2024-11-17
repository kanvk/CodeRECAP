import streamlit as st
import os
from project.indexingUtils import clone_repository, analyze_python_files, index_repo_files_and_functions
from project.azureClient import get_azure_chat_client, get_azure_llm_response, MODEL_NAME
from project.llmPrompts import code_location_template

def locate_files(repo_name, problem_description, hints, files_list):
    # clone repo at commit
    #repo_name, clone_dir = clone_repository(repo_url=repo_url, commit_hash=commit_hash)
    #st.session_state.repo_name = repo_name
    #
    #index_repo_files_and_functions(repo_name, clone_dir)

    # create_database(db_name=repo_name)
    # identify files and functions
    # file_infos, files_list, function_infos = analyze_python_files(clone_dir)
        
    # file_names = [os.path.basename(file_path) for file_path in files_list]
    # IN THE PROMPT FUNCTION INFO IS NOT PASSED TEMPOARARILY
    #print(f"Identified {len(function_infos)} functions and {len(file_infos)} files.")
    chat_client = get_azure_chat_client()
    #if len(function_infos)>1000:
    #    functions_info = f"{len(function_infos)} functions found. Details cannot be listed for each function."
    #else:
    #    functions_info=[
    #        {k: v for k, v in data.items() if k != "code_snippet"}
    #        for data in function_infos
    #    ]
    file_batches = [files_list[i:i + 100] for i in range(0, len(files_list), 100)]

    responses = []
    for batch in file_batches:
        batch_files_list = ", ".join(batch)
        
        query = f"What are the files that need to be modified to solve this problem? Problem description: {problem_description}."
        if hints and hints!="":
            query = f"{query} Hints for solving the problem: {hints}"
        
        prompt = code_location_template.format(
            repo_name=repo_name,
            files_list=batch,
            functions_info=batch_files_list,
            query=query,
        )
        response = get_azure_llm_response(chat_client, MODEL_NAME, prompt)
        responses.append(response)
    
    return responses

def sweBenchClone(repo_url, commit_hash):
    # clone repo at commit
    repo_name, clone_dir = clone_repository(repo_url=repo_url, commit_hash=commit_hash)
    # identify files and functions
    file_infos, files_list, function_infos = analyze_python_files(clone_dir)
        
    # file_names = [os.path.basename(file_path) for file_path in files_list]
    print(f"Identified {len(function_infos)} functions and {len(file_infos)} files.")
    
    return files_list, file_infos, function_infos
    
def sweBenchQuery(files_list, repo_name, problem_description, hints):
    # IN THE PROMPT FUNCTION INFO IS NOT PASSED TEMPORARILY
    chat_client = get_azure_chat_client()
    file_batches = [files_list[i:i + 100] for i in range(0, len(files_list), 100)]
    responses = []
    for batch in file_batches:
        batch_files_list = ", ".join(batch)
        
        query = f"What are the files that need to be modified to solve this problem? Problem description: {problem_description}."
        if hints and hints!="":
            query = f"{query} Hints for solving the problem: {hints}"
        
        prompt = code_location_template.format(
            repo_name=repo_name,
            files_list=batch,
            functions_info=batch_files_list,
            query=query,
        )
        response = get_azure_llm_response(chat_client, MODEL_NAME, prompt)
        responses.append(response)
    
    return responses


if __name__=="__main__":
    streamlit_log = False
    repo_url = "https://github.com/kanvk/CodeRECAP"
    commit_hash = "7ba317a7a6db20d67a37e0eb702af0e36ed2f5bf"
    problem_description = "Update the azure model used for llm calls."
    hints = None
    print("The files to be modified are:",locate_files(repo_url,commit_hash,problem_description,hints))