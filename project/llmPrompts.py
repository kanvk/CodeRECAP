from langchain.prompts import PromptTemplate

function_summary_template = PromptTemplate(
    input_variables=["function_name", "function_code"],
    template="""
        Summarize the following Python function:

        Function Name: {function_name}

        Function Code:
        {function_code}

        Provide a brief summary that describes:
        1. The purpose of the function.
        2. The inputs it takes, including parameter types.
        3. The output it returns.
        4. Any specific logic or steps it follows to achieve its result."""
)

file_summary_template = PromptTemplate(
    input_variables=["file_name", "file_code"],
    template="""
        Summarize the following Python source code file:

        File Name: {file_name}

        File Code:
        {file_code}

        Provide a concise summary that describes:
        1. The overall purpose of the file.
        2. The main classes, functions, or modules defined within it, along with their roles.
        3. Important dependencies or imports and how they are used.
        4. Any notable logic, algorithms, or specific steps followed in the code.
        5. If applicable, how this file interacts with or depends on other files in the project.
    """
)

file_summary_from_function_summary_template = PromptTemplate(
    input_variables=["file_name", "file_code", "function_summaries"],
    template="""
        Summarize the following Python file

        File Name: {file_name}
        File Code:
        {file_code}

        Here are the summaries of each function that is found in this file:
        {function_summaries}

        Provide a concise summary that describes:
        1. The overall purpose of the file.
        2. The main classes, functions, or modules defined within it, along with their roles.
        3. Important dependencies or imports and how they are used.
        4. Any notable logic, algorithms, or specific steps followed in the code.
        5. If applicable, how this file interacts with or depends on other files in the project."""
)

repo_summary_from_file_summary_template = PromptTemplate(
    input_variables=["repo_name", "files_contained", "file_summaries"],
    template="""
        Summarize the following repository

        Repository Name: {repo_name}
        Files found in the repository:
        {files_contained}

        Here are the summaries of the Python files that are found in this repository:
        {file_summaries}

        Provide a concise summary that describes:
        1. The overall purpose of the repository.
        2. The main classes, functions, or modules defined within it, along with their roles.
        3. Any notable logic, algorithms, or specific steps followed in the code.
    """
)

code_location_template = PromptTemplate(
    input_variables=["repo_name", "files_list",
                     "files_info", "functions_info"],
    template="""
        The repository {repo_name} contains the following files:
        {files_list}
        
        This is the information extracted from the repository:
        
        Details on the functions: {functions_info}
        
        Your task is to answer the following user query:
        {query}

        MOST IMPORTANT:
        1. Do not make up file names. The returned file name should be in the above list.
        2. Do not provide an explaination. Just return the file name
        3. if you cannot answer users query reply with "UNABLE TO ANSWER".

    """
)
