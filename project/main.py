import gradio as gr


def hello_world(name):
    return f"Hello, {name}!"


interface = gr.Interface(fn=hello_world, inputs="textbox", outputs="text")

interface.launch()
