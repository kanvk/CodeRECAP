import gradio as gr
from git import Repo


def hello_world(name):
    return f"Hello, {name}!"

def clone_repository(repo_url):
    # Clone the repo with url repo_url into the cloned_repos directory
    # TODO: Check if repo exists and run pull instead of clone if it exists
    repo_name = repo_url.split("/")[-1][:-4]
    clone_dir = f"./cloned_repos/{repo_name}"
    try:
        Repo.clone_from(repo_url, clone_dir)
        status = f"Repository cloned successfully into {clone_dir}"
    except Exception as e:
        status = f"An error occurred while cloning the repository: {e}"
    return status

# interface = gr.Interface(fn=hello_world, inputs="textbox", outputs="text")
interface = gr.Interface(fn=clone_repository, inputs="textbox", outputs="text")

interface.launch()


# Example usage
# repo_url = "https://github.com/kanvk/CodeRECAP.git"