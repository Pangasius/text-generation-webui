"""Module that define the extensions basic functions and calls the llama_index_extension."""
import datetime
import traceback
from typing import List

import torch


from llama_index.schema import (
    MetadataMode,
    NodeWithScore
)

import wandb
from wandb.sdk.data_types.trace_tree import Trace

from modules import shared
from modules.chat import get_turn_substrings
from modules.text_generation import (
    generate_reply_HF,
    generate_reply_custom,
)
from modules.logging_colors import logger

from extensions.llamaindex.llama_index_extension import IndexEngine

DATASET = "f_embed_jira_raw"
INDEX_NAME = "embed_jira_raw"

HISTORY_TREE_SUMMARIZE_TMPL = (
    "\nChat history is above.\n"
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and chat history but not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
)


def setup():
    """
    This function is called once when the extension is loaded.
    After the model has been loaded.
    """
    if shared.index is not None:
        print("Index already loaded!")


    #TODO: Change when go to production
    #wandb.init(project="Haulogy-First-Test")

    shared.index = IndexEngine(index_name=INDEX_NAME, dataset=DATASET).as_retriever(kg=False,
                                            fine_tune=False,
                                            build_index=False)


def get_meta_if_possible(nodes: List[NodeWithScore]) -> str:
    """
    Try to get the title and url of the nodes, if possible.
    Also gives the similarity score if it exists.

    Args:
        nodes (List[NodeWithScore]): List of nodes to get the metadata from
    Returns:
        info (str): String containing the metadata
    """
    info = ""

    for node in nodes:
        if node.metadata is not None:
            # For confluence reader
            if ("url" in node.metadata.keys()
                    and "title" in node.metadata.keys()):
                info += 'Confluence: [' + node.metadata['title'] + '](' + node.metadata['url'] + ')'

            # For jira reader
            if ("key" in node.metadata.keys()
                    and "summary" in node.metadata.keys()):
                info += 'Jira: [' + node.metadata['summary'] + '](' + node.metadata['key'] + ')'
        else:
            # put a small extract instead
            info += 'No metadata found'
        if node.get_score() is not None:
            info += ' Similarity: ' + str(node.get_score())

        info += '\n'

    return info


def input_modifier(question: str, state: dict, is_chat: bool = False) -> str:
    """
    This function is called before the question is sent to the model.
    Adds additional context to the question by asking the retriever
    for similar questions and adding it to the question.

    Args:
        question (str): The original question
        state (dict): The state of the chatbot
        is_chat (bool, optional): Whether the question is part of a chat.
                                  Defaults to False.

    Returns:
        str: The modified question
    """

    state['last_question'] = question

    print("Question: \n", question)

    with torch.no_grad():
        resp = shared.index.retrieve(question)

    context = "\n\n".join([x.get_content(metadata_mode=MetadataMode.NONE)
                            for x in resp])

    state['last_metadata'] = get_meta_if_possible(resp)

    history = get_turn_substrings(state)
    history = "\n".join(history)

    question = HISTORY_TREE_SUMMARIZE_TMPL.format(context_str=context,
                                                    query_str=question)

    print("Question length: ", len(question))
    return question


def output_modifier(output: str, state: dict, is_chat=False):
    """
    This function is called after the model has generated a reply.
    Adds the sources corresponding to the title, url and similarity score.
    In case the metadata is not available, it skips this step.

    Args:
        string (str): the original reply
        state (dict): the state of the chatbot
        is_chat (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if "last_metadata" in state.keys():
        output = output + "\nSources : \n\n" + state["last_metadata"] + "\n\n"

    return output


def custom_generate_reply(*args, **kwargs):
    """
    This function is called when the user presses the "Generate Reply" button.
    Here it acts as a wrapper to log the inputs and outputs of the model using wandb.

    Args:
        question (str): the question
        original_question (str): the original question,
                                 is not the same as question only
                                 for non-chat mode
        seed (int): the generation seed
        state (dict): the state of the chatbot
        stopping_strings (List, optional): List containing the strings to
                                           put an early stop the production.
                                           Defaults to None.
        is_chat (bool, optional): Wether the chatbot is in chat mode.
                                  Defaults to False.

    Yields:
        str: The generated reply
    """

    generate_func = None

    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        return

    if shared.model.__class__.__name__ in ['LlamaCppModel',
                                           'RWKVModel',
                                           'ExllamaModel',
                                           'Exllamav2Model',
                                           'CtransformersModel']:
        generate_func = generate_reply_custom
    else:
        generate_func = generate_reply_HF

    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

    llm_span = Trace(
            name=shared.model_name,
            kind="llm",
            start_time_ms=start_time_ms,
            metadata={"error": "",
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": shared.model_name,
                    "dataset": DATASET,
                    "config": shared.model_config})
    ans = ""

    question = args[0]
    state = args[3]

    state["truncation_length"] = shared.args.n_ctx

    try:
        for ans in generate_func(*args, **kwargs):
            yield ans

    except RuntimeError as e:
        traceback.print_exc()
        llm_span.metadata["error"] = str(e)
    finally:
        last_metadata = state['last_metadata'] if "last_metadata" in state.keys() else ""

        last_question = state['last_question'] if "last_question" in state.keys() else ""

        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        llm_span.inputs = {"query": last_question,
                           "input_documents": last_metadata,
                           "context": question}
        llm_span.end_time_ms = end_time_ms
        llm_span.outputs = {"response": ans}
        llm_span.log(name="HauBot-test-Jira")
