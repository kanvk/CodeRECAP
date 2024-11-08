import streamlit as st
import os
from vectorizeCode import search_vector_store
from azureClient import get_azure_chat_client, get_azure_llm_response, MODEL_NAME
from llmPrompts import code_location_template
from tfidf import search_tfidf


def update_querying_output_log(msg, append=True):
    if append:
        st.session_state["querying_log"] += "  \n" + msg
    else:
        st.session_state["querying_log"] = msg
    print(msg)


def reset_querying_output_log():
    st.session_state["querying_log"] = ""

def display_top_k_similar_docs_tfidf(query, k, docs_type):
    results = search_tfidf(
        query, st.session_state.tfidf_vectorizer, st.session_state.tfidf_matrix, st.session_state.file_infos, k)

    update_querying_output_log("\n"+"-"*25)
    update_querying_output_log(
        f"Top {k} {docs_type}-level matches : (obtained comparing the query against the tf-idf {docs_type} matrix)")
    
    for rank, file_name in enumerate(results):
        update_querying_output_log(f"{rank + 1}. {file_name}")


def display_top_k_similar_docs_tfidf(query, k, docs_type, display=True):
    results = search_tfidf(
        query,
        st.session_state.tfidf_vectorizer,
        st.session_state.tfidf_matrix,
        st.session_state.file_infos,
        k,
    )
    if display:
        update_querying_output_log("\n" + "-" * 25)
        update_querying_output_log(
            f"Top {k} {docs_type}-level matches : (obtained comparing the query against the tf-idf {docs_type} matrix)"
        )
        for rank, file_name in enumerate(results):
            update_querying_output_log(f"{rank + 1}. {file_name}")

    return results

def display_top_k_similar_docs(vector_store, query, k, docs_type, display=True):
    # Takes a vector_store and query as input and returns the top k docs matching the query
    results = search_vector_store(vector_store, query, k)
    if display:
        update_querying_output_log("\n" + "-" * 25)
        update_querying_output_log(
            f"Top {k} {docs_type}-level matches: (obtained comparing the query against the indexed {docs_type} embeddings)"
        )
    for rank, doc in enumerate(results):
        update_querying_output_log(f"{rank+1}. {doc.metadata["name"]}")

    return results


def display_llm_response(query):
    chat_client = get_azure_chat_client()
    prompt = code_location_template.format(
        repo_name=st.session_state.repo_name,
        files_list=st.session_state.files_list,
        files_info=st.session_state.file_infos,
        functions_info=[
            {k: v for k, v in data.items() if k != "code_snippet"}
            for data in st.session_state.function_infos
        ],
        query=query,
    )
    response = get_azure_llm_response(chat_client, MODEL_NAME, prompt)
    update_querying_output_log("\n" + "-" * 25)
    update_querying_output_log(f"LLM Query Response: {response}")
