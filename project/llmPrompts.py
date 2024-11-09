from langchain.prompts import PromptTemplate

function_summary_template = PromptTemplate(
    input_variables=["function_name", "function_code"],
    template="""
        You are given a Python function. Provide a clear and concise summary for this function, focusing on the following aspects:

        Function Name: {function_name}

        Function Code:
        {function_code}

        In your summary, please include:
        1. **Purpose**: Explain what the function is designed to accomplish.
        2. **Inputs**: List all inputs, specifying parameter names, types, and any default values.
        3. **Output**: Describe the output the function returns, including the type and structure (e.g., list, dict).
        4. **Logic/Steps**: Briefly outline the main logic, steps, or calculations the function performs, particularly highlighting any unique techniques, iterations, or conditional statements used.

        Format the summary in a readable, concise paragraph, using professional and clear language suitable for technical documentation.
        """,
)


file_summary_template = PromptTemplate(
    input_variables=["file_name", "file_code"],
    template="""
        You are given the contents of a Python source code file. Summarize the key aspects of this file as follows:

        File Name: {file_name}

        File Code:
        {file_code}

        In your summary, include:
        1. **Purpose**: Describe the primary purpose or functionality this file provides in the project.
        2. **Main Components**: List and briefly describe any primary classes, functions, or modules defined in this file, focusing on their responsibilities and how they contribute to the file's purpose.
        3. **Dependencies and Imports**: Identify the external libraries or modules imported and explain their role in the code.
        4. **Key Logic and Algorithms**: Highlight any significant logic, algorithms, or design patterns implemented, including how they work and why they might be notable.
        5. **Project Interaction**: If relevant, explain how this file interacts with other files in the project, specifying any data it exchanges, functions it calls, or classes it inherits from.

        Structure the summary in clear, concise bullet points or short paragraphs, making it easy for developers to quickly understand the purpose and key elements of this file.
    """,
)


file_summary_from_function_summary_template = PromptTemplate(
    input_variables=["file_name", "file_code", "function_summaries"],
    template="""
        You are given a Python source code file along with summaries of each function it contains. Using this information, provide a high-level summary of the file.

        File Name: {file_name}

        File Code:
        {file_code}

        Function Summaries:
        {function_summaries}

        In your summary, include:
        1. **Overall Purpose**: Describe the main purpose of this file within the project, explaining what it aims to achieve or the functionality it provides.
        2. **Primary Components**: Identify and briefly describe any main classes, functions, or modules in the file, including their roles and interconnections, especially based on the provided function summaries.
        3. **Dependencies and Imports**: Mention any external libraries, packages, or modules imported in this file, explaining their relevance and usage in the code.
        4. **Key Logic and Algorithms**: Summarize any notable algorithms, logic, or structural patterns, including how they work and contribute to the file’s functionality.
        5. **Project Integration**: If relevant, outline how this file interacts with other parts of the project or other files, focusing on function calls, data exchanges, or inheritance that might be involved.

        Structure your response in a clear, concise format (preferably bullet points or short paragraphs), allowing developers to quickly understand this file's purpose, components, and relationships within the codebase.
    """,
)


repo_summary_from_file_summary_template = PromptTemplate(
    input_variables=["repo_name", "files_contained", "file_summaries"],
    template="""
        You are given the details of a Python repository, including a list of files it contains and individual summaries for each file. Provide a high-level summary of the repository.

        Repository Name: {repo_name}

        Files in the Repository:
        {files_contained}

        Summaries of Each Python File:
        {file_summaries}

        In your summary, include:
        1. **Overall Purpose**: Describe the main purpose of this repository, explaining the functionality it provides or the problem it addresses.
        2. **Primary Components**: Identify the main classes, functions, or modules across the files in the repository, detailing their roles and how they contribute to the repository's purpose, based on the provided file summaries.
        3. **Notable Logic and Algorithms**: Highlight any significant logic, algorithms, or design patterns present in the code, particularly if they’re central to the repository’s functionality or problem-solving approach.
        4. **Project Structure and Relationships**: Explain any noteworthy structure in the repository, such as dependencies between files, common data flows, or how modules interact with each other to accomplish the project’s goals.

        Present your response in a structured, concise format, using bullet points or short paragraphs to make it easy to read and understand the key aspects of the repository.
    """,
)


code_location_template = PromptTemplate(
    input_variables=[
        "repo_name",
        "files_list",
        "files_info",
        "functions_info",
        "query",
    ],
    template="""
        You are provided with details about the repository named **{repo_name}**.

        **Files in the Repository:**
        {files_list}
        
        **Function Details and Additional Information:**
        {functions_info}

        **User Query:**
        {query}

        **Your Task**:
        - Identify the exact file name that answers the user query based on the files and functions information provided.
        
        **CRITICAL INSTRUCTIONS**:
        1. **DO NOT create or assume any file names**. Only return a file name if it matches exactly with one from the above list.
        2. **Respond ONLY with the file name**. Avoid explanations, additional text, or clarifications.
        3. **If the query cannot be answered with the provided information**, respond strictly with: "UNABLE TO ANSWER".

        **Response Format**:
        - If a relevant file name is identified, return only the file name (e.g., `main.py`).
        - If unable to determine the answer, respond with "FILE NOT FOUND".

        **IMPORTANT**: Precision is essential; ensure that your answer is concise and follows the format exactly.
    """,
)
