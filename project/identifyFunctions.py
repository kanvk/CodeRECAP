import ast


def get_function_info(code):
    tree = ast.parse(code)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Collect different kinds of parameters
            arguments = []
            varlen_arguments = []
            keyword_arguments = []
            positional_arguments = []
            varlen_keyword_arguments = []
            # Positional-only arguments
            if node.args.posonlyargs:
                positional_arguments.extend(
                    [arg.arg for arg in node.args.posonlyargs])
            # Regular positional or keyword arguments
            arguments.extend([arg.arg for arg in node.args.args])
            # Variable-length positional arguments (*args)
            if node.args.vararg:
                varlen_arguments.append(f"*{node.args.vararg.arg}")
            # Keyword-only arguments (after '*')
            if node.args.kwonlyargs:
                keyword_arguments.extend(
                    [arg.arg for arg in node.args.kwonlyargs])
            # Variable-length keyword arguments (**kwargs)
            if node.args.kwarg:
                varlen_keyword_arguments.append(f"**{node.args.kwarg.arg}")
            code_lines = code.splitlines()
            code_snippet = "\n".join(
                code_lines[node.lineno-1: node.end_lineno])
            # Collect function information
            function_info = {
                'name': node.name,
                'start_line': node.lineno,
                'end_line': node.end_lineno,
                'arguments': arguments,
                'varlen_arguments': varlen_arguments,
                'keyword_arguments': keyword_arguments,
                'positional_arguments': positional_arguments,
                'varlen_keyword_arguments': varlen_keyword_arguments,
                'code_snippet': code_snippet
            }
            functions.append(function_info)
    return functions
