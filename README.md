# CodeRECAP

Repository Context Aware Source Code Summarization

## Pre-requisites

Poetry - this repository requires poetry to install python dependencies.

## Installation

1. Install dependencies using `poetry install`.
2. Activate the installed poetry environment with `poetry shell`.
3. Create a .env file using the format below containing a GitHub token to use for LLM inference.
   - If you do not have a Github Personal Access Token, you can create one by going to https://github.com/settings/tokens and selecting Generate New Token.
   - Note: The "user" scope is enough when creating the token and no other permissions are needed.
4. Run CodeRECAP using `streamlit run project/main.py` inside the activated poetry shell.

### Template for .env

```plaintext
AZURE_INFERENCE_CREDENTIAL=TOKEN_HERE
```

## Usage

1. Enter the URL to the git repo to be analyzed and click on the "Index" button.
   - The repo will be cloned and indexed. This step may take a few minutes the first time it is run.
2. Once the indexing is complete, enter your query in the next input box and click on the "Query Now" button.
   - Top 5 function-level matches and top 5 file-level matches will be displayed. These are based on the vector similarity to the indexed functions' and files' code.
   - The query will also be sent to an LLM and its response is displayed.

## Testing
1. Enter the number of rows to be tested from the swebench dataset (maximum number avilable is 500)
2. Press "Test" button
3. Results are saved automatically to "swebench_predictions.csv"
4. A pie chart and a summary of results are diplayed for each method available