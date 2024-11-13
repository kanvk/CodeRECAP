import streamlit as st
import os
from indexingUtils import clone_repository, analyze_python_files
from azureClient import get_azure_chat_client, get_azure_llm_response, MODEL_NAME
from llmPrompts import code_location_template

def locate_files(repo_url, commit_hash, problem_description, hints):
    # clone repo at commit
    repo_name, clone_dir = clone_repository(repo_url=repo_url, commit_hash=commit_hash)
    # create_database(db_name=repo_name)
    # identify files and functions
    file_infos, files_list, function_infos = analyze_python_files(clone_dir)
    file_names = [os.path.basename(file_path) for file_path in files_list]

    print(f"Identified {len(function_infos)} functions and {len(file_infos)} files.")
    chat_client = get_azure_chat_client()
    if len(function_infos)>1000:
        functions_info = f"{len(function_infos)} functions found. Details cannot be listed for each function."
    else:
        functions_info=[
            {k: v for k, v in data.items() if k != "code_snippet"}
            for data in function_infos
        ]
    query = f"What are the files that need to be modified to solve this problem? Problem description: {problem_description}."
    if hints and hints!="":
        query = f"{query} Hints for solving the problem: {hints}"
    prompt = code_location_template.format(
        repo_name=repo_name,
        files_list=file_names,
        functions_info=functions_info,
        query=query,
    )
    response = get_azure_llm_response(chat_client, MODEL_NAME, prompt)
    return response


if __name__=="__main__":
    streamlit_log = False
    repo_url = "https://github.com/kanvk/CodeRECAP"
    commit_hash = "7ba317a7a6db20d67a37e0eb702af0e36ed2f5bf"
    problem_description = "Update the azure model used for llm calls."
    hints = None
    print("The files to be modified are:",locate_files(repo_url,commit_hash,problem_description,hints))